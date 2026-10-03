#!/usr/bin/env python3
"""Preview or copy explicitly selected local notes into this Jekyll repository.

The JSON manifest is an allowlist. This utility never modifies Notes, deletes
files, commits, or pushes. Only --apply writes, after the complete plan validates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from dataclasses import dataclass, field
from urllib.parse import quote, unquote, urlsplit

try:
    import yaml
except ImportError:
    sys.exit(
        "Missing PyYAML. Install with: python3 -m pip install -r "
        "bin/requirements-notes-sync.txt"
    )


class SyncError(Exception):
    """An input failed validation; no planned output should be applied."""


@dataclass
class Entry:
    source: str
    target: str
    slug: str
    enabled: bool
    metadata: dict
    prefix: str = ""
    post_data: dict = field(default_factory=dict)
    existing: bool = False


@dataclass
class Output:
    path: Path
    content: bytes
    kind: str
    status: str


@dataclass
class Plan:
    repo: Path
    outputs: dict[Path, Output] = field(default_factory=dict)
    posts: list[tuple[str, str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    inputs: set[Path] = field(default_factory=set)

    def protect(self, path: Path) -> None:
        resolved = path.resolve()
        if any(output.path.resolve() == resolved for output in self.outputs.values()):
            raise SyncError(f"An output would overwrite a source note or resource: {path}")
        self.inputs.add(resolved)

    def add(self, path: Path, content: bytes, kind: str) -> str:
        if path.resolve() in self.inputs:
            raise SyncError(f"An output would overwrite a source note or resource: {path}")
        old = self.outputs.get(path)
        if old is not None and old.content != content:
            raise SyncError(f"Conflicting output content: {path.relative_to(self.repo)}")
        if path.exists():
            if not path.is_file():
                raise SyncError(f"Output is not a regular file: {path}")
            status = "unchanged" if path.read_bytes() == content else "changed"
        else:
            status = "new"
        self.outputs[path] = Output(path, content, kind, status)
        return status

    def apply(self) -> None:
        if self.errors:
            raise SyncError("Validation failed; no files were written.")
        for output in self.outputs.values():
            if output.status == "unchanged":
                continue
            # Check again immediately before writing, including ancestor symlinks.
            checked_path(output.path, self.repo, "output")
            if output.path.resolve() in self.inputs:
                raise SyncError(f"An output would overwrite a source note or resource: {output.path}")
            output.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".notes-sync-", dir=output.path.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(output.content)
                    handle.flush()
                    os.fsync(handle.fileno())
                mode = output.path.stat().st_mode & 0o777 if output.path.exists() else 0o644
                os.chmod(temporary, mode)
                os.replace(temporary, output.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)


POST_NAME = re.compile(r"_posts/(\d{4}-\d{2}-\d{2})-([a-z0-9][a-z0-9-]*)\.md\Z")
LIQUID_URL = re.compile(r"\{\{\s*(['\"])(.*?)\1\s*\|\s*relative_url\s*\}\}", re.S)
FRONT_START = re.compile(r"\A---[ \t]*\r?\n")
FRONT_END = re.compile(r"^(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.M)
REMOTE_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def checked_path(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise SyncError(f"{description} escapes allowed directory: {path}")
    return resolved


def frontmatter(text: str, description: str, required: bool = False) -> tuple[str, dict, str]:
    start = FRONT_START.match(text)
    if not start:
        if required:
            raise SyncError(f"{description}: missing Jekyll YAML frontmatter")
        return "", {}, text
    end = FRONT_END.search(text, start.end())
    if end is None:
        raise SyncError(f"{description}: unterminated YAML frontmatter")
    try:
        data = yaml.safe_load(text[start.end() : end.start()]) or {}
    except yaml.YAMLError as exc:
        raise SyncError(f"{description}: invalid YAML frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        raise SyncError(f"{description}: YAML frontmatter must be an object")
    return text[: end.end()], data, text[end.end() :]


def validate_date(value: object, description: str) -> None:
    if not isinstance(value, (str, dt.date, dt.datetime)):
        raise SyncError(f"{description}: date must be an ISO date or timestamp")
    try:
        if isinstance(value, str):
            if len(value) == 10:
                dt.date.fromisoformat(value)
            else:
                dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SyncError(f"{description}: invalid date {value!r}") from exc


def validate_metadata(data: dict, description: str) -> None:
    if not isinstance(data.get("title"), str) or not data["title"].strip():
        raise SyncError(f"{description}: nonempty metadata.title is required")
    validate_date(data.get("date"), description)
    if "published" in data and not isinstance(data["published"], bool):
        raise SyncError(f"{description}: published must be true or false")
    permalink = data.get("permalink")
    if permalink is not None and (
        not isinstance(permalink, str)
        or not permalink.startswith("/")
        or permalink.startswith("//")
        or any(c in permalink for c in "?#{}\\")
        or ".." in PurePosixPath(permalink).parts
    ):
        raise SyncError(f"{description}: permalink must be a plain site path beginning with /")


def parse_entries(config: dict, repo: Path) -> list[Entry]:
    if not isinstance(config, dict) or config.get("version") != 1:
        raise SyncError("Config must be a JSON object with version: 1")
    if not isinstance(config.get("posts"), list):
        raise SyncError("Config posts must be an array")
    sources, targets = set(), set()
    entries = []
    posts_root = checked_path(repo / "_posts", repo, "posts directory")
    for index, item in enumerate(config["posts"], start=1):
        if not isinstance(item, dict):
            raise SyncError(f"Config posts[{index}]: entry must be an object")
        source, target = item.get("source"), item.get("target")
        if not isinstance(source, str) or not source:
            raise SyncError(f"Config posts[{index}]: source must be a relative Markdown path")
        source_path = PurePosixPath(source)
        if (
            source_path.is_absolute()
            or ".." in source_path.parts
            or "\\" in source
            or str(source_path) != source
            or source_path.suffix.lower() not in (".md", ".markdown")
        ):
            raise SyncError(f"Invalid relative source path: {source!r}")
        match = POST_NAME.fullmatch(target) if isinstance(target, str) else None
        if match is None:
            raise SyncError(f"Invalid target: {target!r}; expected _posts/YYYY-MM-DD-slug.md")
        validate_date(match[1], target)
        checked_path(repo / target, posts_root, "post target")
        if source in sources or target in targets:
            raise SyncError(f"Duplicate source or target in config: {source} -> {target}")
        sources.add(source)
        targets.add(target)
        enabled = item.get("enabled", True)
        metadata = item.get("metadata", {})
        if not isinstance(enabled, bool) or not isinstance(metadata, dict):
            raise SyncError(f"{source}: enabled must be boolean and metadata must be an object")
        entries.append(Entry(source, target, match[2], enabled, metadata))
    return entries


def prepare_metadata(entry: Entry, repo: Path) -> None:
    target = repo / entry.target
    entry.existing = target.exists()
    if entry.existing:
        if not target.is_file():
            raise SyncError(f"Post target is not a regular file: {entry.target}")
        entry.prefix, entry.post_data, _ = frontmatter(
            target.read_bytes().decode("utf-8"), entry.target, required=True
        )
        if entry.post_data.get("source_path") != entry.source:
            raise SyncError(f"{entry.target}: source_path does not match {entry.source!r}")
    else:
        data = dict(entry.metadata)
        if "source_path" in data and data["source_path"] != entry.source:
            raise SyncError(f"{entry.source}: metadata.source_path must match source")
        data.setdefault("layout", "post")
        data.setdefault("categories", [])
        data.setdefault("permalink", f"/blog/{entry.slug}/")
        data["notes_import"] = True
        data["source_path"] = entry.source
        entry.post_data = data
        entry.prefix = "---\n" + yaml.safe_dump(data, allow_unicode=True, sort_keys=False) + "---\n"
    validate_metadata(entry.post_data, entry.target)


class Transformer:
    def __init__(self, plan: Plan, root: Path, entry: Entry, source: Path, links: dict[Path, str]):
        self.plan, self.root, self.entry, self.source, self.links = plan, root, entry, source, links
        self.asset_root = checked_path(plan.repo / "assets/blog", plan.repo, "asset directory")
        self.asset_sources: dict[Path, Path] = {}

    def asset(self, path: Path, forced_destination: Path | None = None) -> str:
        source = checked_path(path, self.root, "referenced resource")
        if not source.is_file():
            raise SyncError(f"Missing referenced resource: {path}")
        if source.suffix.lower() in (".md", ".markdown", ".mdown"):
            raise SyncError(f"Markdown notes are never copied as attachments: {path}")
        self.plan.protect(source)
        if forced_destination is None:
            if source.is_relative_to(self.source.parent):
                relative = source.relative_to(self.source.parent)
            else:
                relative = Path("_shared") / source.relative_to(self.root)
            destination = self.asset_root / self.entry.slug / relative
            previous = self.asset_sources.get(destination)
            if previous is not None and previous != source:
                digest = hashlib.sha256(str(source.relative_to(self.root)).encode()).hexdigest()[:12]
                destination = destination.with_name(f"{destination.stem}-{digest}{destination.suffix}")
        else:
            destination = forced_destination
        checked_path(destination, self.asset_root, "asset target")
        self.asset_sources[destination] = source
        self.plan.add(destination, source.read_bytes(), "asset")
        encoded = quote("/" + destination.relative_to(self.plan.repo).as_posix(), safe="/")
        return "{{ '" + encoded + "' | relative_url }}"

    def url(self, original: str, image: bool = False, wiki: bool = False) -> str:
        liquid = LIQUID_URL.fullmatch(original)
        if liquid:
            route = unquote(liquid[2])
            prefix = f"/assets/blog/{self.entry.slug}/"
            if route.startswith(prefix):
                suffix = route[len(prefix) :]
                if ".." in PurePosixPath(suffix).parts:
                    raise SyncError(f"Liquid resource traverses its blog directory: {route}")
                destination = self.plan.repo / route.lstrip("/")
                checked_path(destination, self.asset_root, "Liquid resource target")
                local = checked_path(self.source.parent / suffix, self.root, "Liquid resource source")
                if local.is_file() and local.suffix.lower() not in (".md", ".markdown", ".mdown"):
                    self.asset(local, forced_destination=destination)
                elif not destination.is_file():
                    raise SyncError(f"Missing existing Liquid resource: {route}")
            return original
        if "{{" in original or "{%" in original:
            # Other intentional Jekyll expressions are already website references.
            return original
        if original.startswith(("#", "//")) or not original:
            return original
        raw = re.sub(r"\\([\\ ()\[\]])", r"\1", original)
        if re.match(r"^[A-Za-z]:[\\/]", raw) or raw.startswith("\\\\"):
            raise SyncError(f"Windows resource paths are unsupported; convert to a note-relative path: {original}")
        parsed = urlsplit(raw)
        if REMOTE_SCHEME.match(raw) and parsed.scheme != "file":
            return original
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            raise SyncError(f"Unsupported file URL host: {original}")
        path_text = unquote(parsed.path)
        if not path_text:
            return original
        if path_text.startswith("/") and parsed.scheme != "file":
            local_absolute = path_text.startswith(("/Users/", "/home/", "/tmp/", str(self.root) + "/"))
            if not local_absolute:
                return original  # Existing website paths, including /assets/.
        path = Path(path_text)
        if wiki and not image and not path.suffix:
            path = path.with_suffix(".md")
        local = checked_path(path if path.is_absolute() else self.source.parent / path, self.root, "local link")
        tail = ("?" + parsed.query if parsed.query else "") + ("#" + parsed.fragment if parsed.fragment else "")
        if local.suffix.lower() in (".md", ".markdown", ".mdown"):
            if image:
                raise SyncError(f"Cannot embed a Markdown note as an image: {original}")
            permalink = self.links.get(local)
            if permalink is None:
                raise SyncError(f"Local note is not an enabled, published blog: {original}")
            encoded = quote(permalink, safe="/")
            return "{{ '" + encoded + "' | relative_url }}" + tail
        return self.asset(local) + tail

    def transform(self, body: str) -> str:
        # Placeholders keep complete links containing inline-code labels parseable,
        # while all code content is restored without any transformation.
        marker = "\x00NOTES_SYNC_CODE_"
        if marker in body:
            raise SyncError("Source contains a reserved synchronization marker")
        substitutions, pieces = [], []
        for protected, text in protected_segments(body):
            if protected and text:
                token = f"{marker}{len(substitutions)}\x00" + ("\n" if text.endswith("\n") else "")
                substitutions.append((token, text))
                pieces.append(token)
            else:
                pieces.append(text)
        transformed = self.markup("".join(pieces))
        for token, text in substitutions:
            transformed = transformed.replace(token, text)
        return transformed

    def markup(self, text: str) -> str:
        # HTML destinations are quoted so spaces and Liquid expressions stay intact.
        html = re.compile(r"(<(?:img|a|source|video|audio)\b[^>]*?\b(?:src|href|poster)\s*=\s*)([\"'])(.*?)\2", re.I | re.S)
        text = html.sub(lambda m: m[1] + m[2] + self.url(m[3], image=m[1].lower().startswith("<img")) + m[2], text)
        if re.search(r"<img\b[^>]*\bsrc\s*=\s*[^\s\"'>]", text, re.I):
            raise SyncError("HTML img src must be quoted for safe resource synchronization")
        if re.search(r"<(?:img|source)\b[^>]*\bsrcset\s*=", text, re.I):
            raise SyncError("HTML srcset is unsupported; use individual quoted src attributes")
        # Reference definitions use the same destination grammar as inline links.
        def reference(match: re.Match) -> str:
            line = match[0]
            start = match.end(1) - match.start()
            destination = destination_span(line, start)
            if destination is None:
                raise SyncError(f"Unsupported Markdown reference destination: {line.strip()}")
            first, last = destination
            return line[:first] + self.url(line[first:last]) + line[last:]
        text = re.sub(r"^([ \t]{0,3}\[(?!\^)[^\]\n]+\]:[ \t]*).*", reference, text, flags=re.M)
        chunks, cursor, index = [], 0, 0
        while index < len(text):
            if text[index] == "\\":
                index += 2
                continue
            image = text.startswith("![", index)
            opener = index + 1 if image else index
            if text.startswith("[[", opener):
                close = text.find("]]", opener + 2)
                if close == -1:
                    raise SyncError("Unclosed Obsidian wiki link; expected ]]")
                value = text[opener + 2 : close]
                if "\n" in value or value.count("|") > 1:
                    raise SyncError(f"Unsupported Obsidian wiki syntax: {value}")
                parts = value.split("|", 1)
                if image and len(parts) == 2 and re.fullmatch(r"\d+(?:x\d+)?", parts[1]):
                    raise SyncError("Obsidian image dimensions are unsupported; use Markdown or HTML")
                label = parts[1] if len(parts) == 2 else ("" if image else Path(parts[0]).stem)
                label = label.replace("[", "\\[").replace("]", "\\]")
                replacement = ("!" if image else "") + "[" + label + "](" + self.url(parts[0], image=image, wiki=True) + ")"
                chunks.extend((text[cursor:index], replacement))
                cursor = index = close + 2
                continue
            if opener < len(text) and text[opener] == "[":
                close = bracket_end(text, opener)
                if close is not None and close + 1 < len(text) and text[close + 1] == "(":
                    destination = destination_span(text, close + 2)
                    if destination is not None:
                        first, last = destination
                        replacement = self.url(text[first:last], image=image)
                        chunks.extend((text[cursor:first], replacement))
                        cursor = index = last
                        continue
            index += 1
        chunks.append(text[cursor:])
        return "".join(chunks)


def bracket_end(text: str, start: int) -> int | None:
    depth, index = 1, start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def destination_span(text: str, start: int) -> tuple[int, int] | None:
    index = start
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    if index == len(text):
        return None
    if text[index] == "<":
        end = text.find(">", index + 1)
        return (index + 1, end) if end != -1 else None
    begin, depth = index, 0
    if text.startswith("{{", index):
        end = text.find("}}", index + 2)
        if end == -1:
            return None
        index = end + 2
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif char.isspace() and depth == 0:
            break
        index += 1
    return (begin, index) if index > begin else None


def protected_segments(text: str):
    """Keep fenced, ordinary indented, and inline backtick code unchanged."""
    spans, offset, fence_start, fence_char, fence_length = [], 0, None, "", 0
    indented_start, code_indent, list_indent, previous_blank = None, 4, None, True
    for line in text.splitlines(keepends=True):
        fence = re.match(r"^[ \t]*(?:>[ \t]*)*(`{3,}|~{3,})(.*)", line)
        if fence_start is None and fence:
            if indented_start is not None:
                spans.append((indented_start, offset))
                indented_start = None
            fence_start, fence_char, fence_length = offset, fence[1][0], len(fence[1])
        elif fence_start is not None and fence and fence[1][0] == fence_char and len(fence[1]) >= fence_length and not fence[2].strip():
            spans.append((fence_start, offset + len(line)))
            fence_start = None
            previous_blank = True
        elif fence_start is None:
            expanded = line.expandtabs(4)
            indent = len(expanded) - len(expanded.lstrip(" "))
            blank = not line.strip()
            if indented_start is not None and not blank and indent < code_indent:
                spans.append((indented_start, offset))
                indented_start = None
            if indented_start is None:
                if list_indent is not None and not blank and indent < list_indent:
                    list_indent = None
                threshold = 4 if list_indent is None else list_indent + 4
                if previous_blank and not blank and indent >= threshold:
                    indented_start, code_indent = offset, threshold
                else:
                    marker = re.match(r"^ *(?:[-+*]|\d+[.)])[ \t]+", expanded)
                    if marker:
                        list_indent = marker.end()
            previous_blank = blank
        offset += len(line)
    if fence_start is not None:
        spans.append((fence_start, len(text)))
    if indented_start is not None:
        spans.append((indented_start, len(text)))
    outer_cursor = 0
    for first, last in spans + [(len(text), len(text))]:
        plain = text[outer_cursor:first]
        cursor = 0
        for ticks in re.finditer(r"`+", plain):
            if ticks.start() < cursor:
                continue
            closing = re.search(r"(?<!`)" + re.escape(ticks[0]) + r"(?!`)", plain[ticks.end() :])
            if closing is None:
                continue
            end = ticks.end() + closing.end()
            yield False, plain[cursor : ticks.start()]
            yield True, plain[ticks.start() : end]
            cursor = end
        yield False, plain[cursor:]
        yield True, text[first:last]
        outer_cursor = last


def create_plan(repo: Path, config: dict, notes_root: str | Path | None = None, only: list[str] | None = None) -> Plan:
    repo = repo.resolve()
    entries = parse_entries(config, repo)
    root_value = notes_root if notes_root is not None else config.get("notes_root")
    if not isinstance(root_value, (str, Path)) or not str(root_value):
        raise SyncError("notes_root is required in config or via --notes-root")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise SyncError(f"Notes directory does not exist: {root}; use --notes-root to override")
    requested = set(only or [])
    by_source = {entry.source: entry for entry in entries}
    unknown = requested - by_source.keys()
    if unknown:
        raise SyncError("--only source not in manifest: " + ", ".join(sorted(unknown)))
    disabled = sorted(source for source in requested if not by_source[source].enabled)
    if disabled:
        raise SyncError("--only source is disabled: " + ", ".join(disabled))
    selected = [entry for entry in entries if entry.enabled and (not requested or entry.source in requested)]
    selected_sources = {entry.source for entry in selected}
    plan, links, ready, permalinks, asset_buckets = Plan(repo), {}, set(), {}, {}
    for entry in entries:
        plan.protect(root / entry.source)
    for entry in entries:
        if not entry.enabled:
            continue
        try:
            # Missing unselected source files are allowed; an existing public post
            # remains a valid link target when synchronizing just another note.
            source = checked_path(root / entry.source, root, "source note")
            prepare_metadata(entry, repo)
            ready.add(entry.source)
            if (entry.existing or entry.source in selected_sources) and entry.post_data.get("published", True):
                permalink = entry.post_data.get("permalink", f"/blog/{entry.slug}/")
                normalized = unquote(permalink).removesuffix("index.html").rstrip("/")
                if normalized in permalinks:
                    raise SyncError(f"Duplicate published permalink {permalink!r}: {permalinks[normalized]} and {entry.target}")
                if entry.slug in asset_buckets:
                    raise SyncError(f"Duplicate blog asset directory for slug {entry.slug!r}: choose unique target slugs")
                permalinks[normalized] = entry.target
                asset_buckets[entry.slug] = entry.target
                links[source] = permalink
        except (SyncError, OSError, UnicodeError) as exc:
            plan.errors.append(f"{entry.source}: {exc}")
    for entry in selected:
        if entry.source not in ready:
            continue
        try:
            source = checked_path(root / entry.source, root, "source note")
            if not source.is_file():
                raise SyncError(f"Missing source note: {source}")
            _, _, body = frontmatter(source.read_bytes().decode("utf-8-sig"), entry.source)
            body = Transformer(plan, root, entry, source, links).transform(body)
            content = entry.prefix + "\n" + body.lstrip("\r\n")
            if not content.endswith("\n"):
                content += "\n"
            status = plan.add(repo / entry.target, content.encode("utf-8"), "post")
            plan.posts.append((entry.source, entry.target, status))
        except (SyncError, OSError, UnicodeError) as exc:
            plan.errors.append(f"{entry.source}: {exc}")
    return plan


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo / "notes-sync.json", help="JSON allowlist (default: repository notes-sync.json)")
    parser.add_argument("--notes-root", type=Path, help="Override the local Notes directory")
    parser.add_argument("--only", action="append", default=[], metavar="SOURCE", help="Exact manifest source path; repeat to select multiple notes")
    parser.add_argument("--apply", action="store_true", help="Write after validation; without this flag, only preview")
    parser.add_argument("--diff", action="store_true", help="Show unified Markdown post diffs in addition to the preview summary")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        plan = create_plan(repo, config, notes_root=args.notes_root, only=args.only)
        for source, target, status in plan.posts:
            print(f"{status.upper():9} {source} -> {target}")
        for output in plan.outputs.values():
            if output.kind == "asset":
                print(f"ASSET {output.status.upper():9} {output.path.relative_to(repo)}")
            elif args.diff and output.status != "unchanged":
                before = output.path.read_text(encoding="utf-8") if output.path.exists() else ""
                filename = output.path.relative_to(repo).as_posix()
                diff = difflib.unified_diff(
                    before.splitlines(keepends=True),
                    output.content.decode("utf-8").splitlines(keepends=True),
                    fromfile="a/" + filename,
                    tofile="b/" + filename,
                )
                sys.stdout.writelines(diff)
        changed = sum(output.kind == "post" and output.status != "unchanged" for output in plan.outputs.values())
        assets = sum(output.kind == "asset" and output.status != "unchanged" for output in plan.outputs.values())
        print(f"\n{'Apply' if args.apply else 'Preview'}: {len(plan.posts)} posts checked; {changed} posts and {assets} assets would change.")
        if plan.errors:
            for error in plan.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            print("Validation failed; no files were written.", file=sys.stderr)
            return 1
        if args.apply:
            plan.apply()
            print("Sync complete. Review git diff before committing and pushing.")
        else:
            print("No files were written. Add --apply to synchronize these changes.")
        return 0
    except (SyncError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}\nNo further files were written.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
