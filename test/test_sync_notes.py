"""Safety and publication tests for the local Notes synchronizer.

Run: python3 -m unittest discover -s test -p 'test_sync_notes.py' -v
All notes and repository outputs in these tests are temporary fixtures.
"""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "bin/sync_notes.py"
SPEC = importlib.util.spec_from_file_location("sync_notes", SCRIPT)
sync = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync
SPEC.loader.exec_module(sync)


class NotesSyncTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo, self.notes = self.base / "repo", self.base / "Notes"
        self.repo.mkdir()
        self.notes.mkdir()
        self.config = {"version": 1, "notes_root": str(self.notes), "posts": []}

    def write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        return path

    def entry(self, source="topic/main.md", slug="main", enabled=True, metadata=None):
        item = {
            "source": source,
            "target": f"_posts/2026-10-03-{slug}.md",
            "metadata": metadata or {"title": slug, "date": "2026-10-03"},
            "enabled": enabled,
        }
        self.config["posts"].append(item)
        return item

    def existing(self, entry, extra="", body="old body\n", newline="\n"):
        prefix = (
            "---\n"
            "# Preserve handwritten YAML comments and key order\n"
            "layout: post\n"
            "title: 'Original title'\n"
            "date: 2026-10-03\n"
            f"source_path: {json.dumps(entry['source'], ensure_ascii=False)}\n"
            f"permalink: /blog/{Path(entry['target']).stem[11:]}/\n"
            "toc:\n  beginning: true\n"
            + extra
            + "---\n"
        ).replace("\n", newline)
        self.write(self.repo / entry["target"], prefix + newline + body)
        return prefix.encode("utf-8")

    def plan(self, **kwargs):
        return sync.create_plan(self.repo, self.config, **kwargs)

    def assert_valid(self, plan):
        self.assertEqual(plan.errors, [])

    def test_cli_preview_diff_never_writes_and_reports_new_outputs(self):
        entry = self.entry()
        self.write(self.notes / entry["source"], "# Body\n![image](pic.png)\n")
        self.write(self.notes / "topic/pic.png", b"image-data")
        script = self.repo / "bin/sync_notes.py"
        script.parent.mkdir()
        shutil.copyfile(SCRIPT, script)
        self.write(self.repo / "notes-sync.json", json.dumps(self.config))
        before = {p.relative_to(self.base): p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        result = subprocess.run([sys.executable, str(script), "--diff"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Preview: 1 posts checked", result.stdout)
        self.assertIn("assets/blog/main/pic.png", result.stdout)
        self.assertIn("+++ b/_posts/2026-10-03-main.md", result.stdout)
        after = {p.relative_to(self.base): p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.repo / "assets").exists())

    def test_apply_idempotence_exact_frontmatter_and_source_preservation(self):
        entry = self.entry(metadata={"title": "Ignored edit", "date": "2000-01-01"})
        prefix = self.existing(entry, newline="\r\n")
        note = b"---\ntitle: Private local title\nprivate: true\n---\n\n# Updated body\n![image](pic.png)\n"
        self.write(self.notes / entry["source"], note)
        self.write(self.notes / "topic/pic.png", b"new-image")
        stale = self.write(self.repo / "assets/blog/main/old.png", b"retained")
        plan = self.plan()
        self.assert_valid(plan)
        plan.apply()
        output = (self.repo / entry["target"]).read_bytes()
        self.assertTrue(output.startswith(prefix))
        self.assertIn(b"# Updated body", output)
        self.assertNotIn(b"Private local title", output)
        self.assertEqual((self.notes / entry["source"]).read_bytes(), note)
        self.assertEqual(stale.read_bytes(), b"retained")
        repeat = self.plan()
        self.assert_valid(repeat)
        self.assertTrue(all(output.status == "unchanged" for output in repeat.outputs.values()))

    def test_only_is_exact_allowlist_and_disabled_or_private_notes_not_copied(self):
        main = self.entry()
        other = self.entry("other.md", "other")
        self.entry("private.md", "private", enabled=False)
        self.write(self.notes / main["source"], "Updated main\n")
        self.write(self.notes / "secret.md", "Not on manifest\n")
        # A missing source outside --only must not block a selected note.
        self.existing(other)
        plan = self.plan(only=[main["source"]])
        self.assert_valid(plan)
        plan.apply()
        self.assertEqual(len(plan.posts), 1)
        self.assertFalse((self.repo / "_posts/2026-10-03-private.md").exists())
        with self.assertRaisesRegex(sync.SyncError, "not in manifest"):
            self.plan(only=["main.md"])
        with self.assertRaisesRegex(sync.SyncError, "disabled"):
            self.plan(only=["private.md"])

    def test_note_links_only_resolve_to_published_blog_and_keep_anchors(self):
        main = self.entry()
        other = self.entry("topic/other.md", "other")
        self.existing(other)
        self.write(self.notes / main["source"], "[Other](other.md#section)\n[With `code` label](other.md)\n")
        plan = self.plan(only=[main["source"]])
        self.assert_valid(plan)
        output = plan.outputs[self.repo / main["target"]].content.decode()
        self.assertIn("[Other]({{ '/blog/other/' | relative_url }}#section)", output)
        self.assertIn("[With `code` label]({{ '/blog/other/' | relative_url }})", output)
        self.existing(other, extra="published: false\n")
        unpublished = self.plan(only=[main["source"]])
        self.assertIn("not an enabled, published blog", unpublished.errors[0])
        self.config["posts"][1]["enabled"] = False
        self.assertTrue(self.plan(only=[main["source"]]).errors)

    def test_unselected_new_note_or_unmanaged_note_is_not_implicitly_published(self):
        main = self.entry()
        self.entry("topic/new.md", "new")
        self.write(self.notes / main["source"], "[New](new.md)\n")
        self.write(self.notes / "topic/new.md", "New body\n")
        plan = self.plan(only=[main["source"]])
        self.assertIn("not an enabled, published blog", plan.errors[0])
        self.assertFalse((self.repo / "_posts").exists())
        self.write(self.notes / main["source"], "[Secret](secret.md)\n")
        self.write(self.notes / "topic/secret.md", "Secret\n")
        self.assertTrue(self.plan(only=[main["source"]]).errors)

    def test_unicode_spaces_nested_resources_and_collisions(self):
        entry = self.entry()
        source = (
            "![Unicode](图片/图%20一.png)\n"
            "![Same name](elsewhere/图%20一.png)\n"
            "[PDF](<files/read me.pdf>)\n"
            "![Shared](../shared/pic.png)\n"
            "![Collision](_shared/shared/pic.png)\n"
        )
        self.write(self.notes / entry["source"], source)
        paths = {
            "topic/图片/图 一.png": b"one",
            "topic/elsewhere/图 一.png": b"two",
            "topic/files/read me.pdf": b"pdf",
            "shared/pic.png": b"external",
            "topic/_shared/shared/pic.png": b"internal",
        }
        for path, content in paths.items():
            self.write(self.notes / path, content)
        plan = self.plan()
        self.assert_valid(plan)
        assets = [out for out in plan.outputs.values() if out.kind == "asset"]
        self.assertEqual(len(assets), 5)
        self.assertEqual({out.content for out in assets}, set(paths.values()))
        output = plan.outputs[self.repo / entry["target"]].content.decode()
        self.assertIn("%E5%9B%BE%20%E4%B8%80.png", output)
        self.assertIn("read%20me.pdf", output)
        plan.apply()
        second = self.plan()
        self.assert_valid(second)
        self.assertTrue(all(out.status == "unchanged" for out in second.outputs.values()))

    def test_reference_html_wiki_and_code_examples(self):
        entry = self.entry()
        other = self.entry("topic/other.md", "other")
        self.existing(other)
        source = (
            "![Ref][image]\n[image]: pic.png \"Image title\"\n"
            '<img src="pic.png" alt="Picture" />\n'
            '<a href="paper.pdf">PDF</a>\n'
            "![[pic.png]]\n[[other.md|Other note]]\n"
            "A footnote[^1].\n[^1]: This is an explanatory footnote.\n"
            "`![Inline example](missing.png)`\n"
            "```markdown\n![Example](missing.png)\n[missing]: nowhere.png\n```\n"
            "> ~~~text\n> ![Example](missing.png)\n> ~~~\n"
            "\n    ![Indented example](missing.png)\n    [Missing](private.md)\n"
            "\n- List item\n\n    ![List attachment](pic.png)\n"
        )
        self.write(self.notes / entry["source"], source)
        self.write(self.notes / "topic/pic.png", b"png")
        self.write(self.notes / "topic/paper.pdf", b"pdf")
        plan = self.plan(only=[entry["source"]])
        self.assert_valid(plan)
        output = plan.outputs[self.repo / entry["target"]].content.decode()
        self.assertIn("[image]: {{ '/assets/blog/main/pic.png' | relative_url }} \"Image title\"", output)
        self.assertIn("[Other note]({{ '/blog/other/' | relative_url }})", output)
        self.assertIn("[^1]: This is an explanatory footnote.", output)
        for example in ("`![Inline example](missing.png)`", "[missing]: nowhere.png", "> ![Example](missing.png)"):
            self.assertIn(example, output)
        self.assertIn("    ![Indented example](missing.png)", output)
        self.assertIn("    ![List attachment]({{ '/assets/blog/main/pic.png' | relative_url }})", output)

    def test_remote_root_urls_and_existing_liquid_resources(self):
        entry = self.entry()
        liquid = "{{ '/assets/blog/main/pic.png' | relative_url }}"
        source = (
            f"![Stored]({liquid})\n"
            "![Remote](https://example.com/pic.png)\n"
            "![Data](data:image/png;base64,AA==)\n"
            "[Anchor](#here)\n"
            "![Site](/assets/img/existing.png)\n"
        )
        self.write(self.notes / entry["source"], source)
        self.write(self.repo / "assets/blog/main/pic.png", b"old")
        self.write(self.notes / "topic/pic.png", b"new")
        plan = self.plan()
        self.assert_valid(plan)
        output = plan.outputs[self.repo / entry["target"]].content.decode()
        self.assertIn(source, output)
        plan.apply()
        self.assertEqual((self.repo / "assets/blog/main/pic.png").read_bytes(), b"new")

    def test_missing_source_or_asset_prevents_all_apply_writes(self):
        present = self.entry()
        self.entry("missing.md", "missing")
        self.write(self.notes / present["source"], "![Picture](pic.png)\n")
        self.write(self.notes / "topic/pic.png", b"planned-image")
        plan = self.plan()
        self.assertIn("Missing source note", plan.errors[0])
        with self.assertRaises(sync.SyncError):
            plan.apply()
        self.assertEqual(list(self.repo.iterdir()), [])
        self.config["posts"].pop()
        self.write(self.notes / present["source"], "![Picture](pic.png)\n![Missing](absent.png)\n")
        missing_asset = self.plan()
        self.assertIn("Missing referenced resource", missing_asset.errors[0])
        with self.assertRaises(sync.SyncError):
            missing_asset.apply()
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_path_traversal_symlinks_and_absolute_resource_escape(self):
        entry = self.entry()
        outside = self.write(self.base / "secret.png", b"secret")
        source = self.notes / entry["source"]
        self.write(source, "![Escape](%2E%2E/%2E%2E/secret.png)\n")
        self.assertIn("escapes allowed directory", self.plan().errors[0])
        self.write(source, "![Escape](escape.png)\n")
        (self.notes / "topic/escape.png").symlink_to(outside)
        self.assertIn("escapes allowed directory", self.plan().errors[0])
        self.write(source, f"![Escape]({outside})\n")
        self.assertIn("escapes allowed directory", self.plan().errors[0])
        source.unlink()
        source.symlink_to(self.write(self.base / "outside.md", "Secret\n"))
        self.assertIn("escapes allowed directory", self.plan().errors[0])

    def test_output_escape_and_source_overlap_never_write(self):
        entry = self.entry()
        self.write(self.notes / entry["source"], "Body\n")
        outside = self.base / "outside-posts"
        outside.mkdir()
        (self.repo / "_posts").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(sync.SyncError, "escapes allowed directory"):
            self.plan()
        (self.repo / "_posts").unlink()
        # A Notes directory may contain a checkout, but an allowlist input must
        # never alias an output within that checkout.
        entry["source"] = entry["target"]
        self.config["notes_root"] = str(self.repo)
        self.existing(entry, body="Private original\n")
        original = (self.repo / entry["target"]).read_bytes()
        overlapping = self.plan()
        self.assertIn("overwrite a source", overlapping.errors[0])
        with self.assertRaises(sync.SyncError):
            overlapping.apply()
        self.assertEqual((self.repo / entry["target"]).read_bytes(), original)

    def test_config_duplicates_wrong_article_and_invalid_new_metadata(self):
        entry = self.entry()
        self.write(self.notes / entry["source"], "Body\n")
        self.config["posts"].append(dict(entry))
        with self.assertRaisesRegex(sync.SyncError, "Duplicate"):
            self.plan()
        self.config["posts"].pop()
        self.existing(entry)
        target = self.repo / entry["target"]
        self.write(target, target.read_bytes().replace(b"topic/main.md", b"wrong.md"))
        self.assertIn("source_path does not match", self.plan().errors[0])
        target.unlink()
        entry["metadata"] = {"title": "New without a date"}
        self.assertIn("date must be", self.plan().errors[0])
        entry["target"] = "_posts/2026-02-30-main.md"
        with self.assertRaisesRegex(sync.SyncError, "invalid date"):
            self.plan()

    def test_duplicate_published_routes_fail_without_writes(self):
        first = self.entry("first.md", "same")
        second = self.entry("second.md", "same")
        second["target"] = "_posts/2026-10-04-same.md"
        self.write(self.notes / first["source"], "First\n")
        self.write(self.notes / second["source"], "Second\n")
        plan = self.plan()
        self.assertIn("Duplicate published permalink", plan.errors[0])
        with self.assertRaises(sync.SyncError):
            plan.apply()
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_windows_resource_paths_error_and_later_assets_are_not_written(self):
        entry = self.entry()
        for path in (r"C:\Users\sebby\Notes\pic.png", "C:/Users/sebby/Notes/pic.png"):
            self.write(self.notes / entry["source"], f"![Old Windows image]({path})\n")
            plan = self.plan()
            self.assertIn("convert to a note-relative path", plan.errors[0])
            with self.assertRaises(sync.SyncError):
                plan.apply()
        self.assertEqual(list(self.repo.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
