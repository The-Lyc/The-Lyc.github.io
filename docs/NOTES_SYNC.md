# 从本地 Notes 同步博客

保留 `/Users/sebby/Documents/Notes` 作为笔记目录，在旁边放一份网站仓库。只同步 `notes-sync.json` 中明确列出的笔记；私人笔记和未列出的目录不会复制到网站。同步后先检查差异，再手动提交、推送。

```text
/Users/sebby/Documents/
├── Notes/                   # 原笔记，继续在这里编辑
└── The-Lyc.github.io/        # 网站仓库，同步输出和 Git 操作都在这里
    ├── notes-sync.json
    ├── _posts/
    └── assets/blog/
```

脚本在你的 Mac 上运行，云端工作区不能直接读取 `/Users/sebby/Documents/Notes`。它只写入网站仓库，不修改原笔记、不删除文章或资源，也不自动执行 Git 命令。

## 首次准备

如果电脑还没有网站仓库，执行：

```bash
git clone https://github.com/The-Lyc/The-Lyc.github.io.git /Users/sebby/Documents/The-Lyc.github.io
cd /Users/sebby/Documents/The-Lyc.github.io
```

已有本地仓库则直接进入该目录，不需要再次克隆。同步工具已随网站仓库提供，更新仓库即可获取：

```bash
cd /Users/sebby/Documents/The-Lyc.github.io
git pull --ff-only
```

需要 Python 3.9 或以上版本。首次创建独立的 Python 环境；已有 `.venv-notes-sync` 则只需执行安装依赖的命令：

```bash
python3 -m venv .venv-notes-sync
.venv-notes-sync/bin/python -m pip install -r bin/requirements-notes-sync.txt
```

可以验证脚本是否正常运行；测试使用临时文件，不读取或发布你的 Notes：

```bash
.venv-notes-sync/bin/python -m unittest discover -s test -p test_sync_notes.py
```

打开 [`notes-sync.json`](../notes-sync.json)，检查 `notes_root` 和每条 `source`。默认根目录是 `/Users/sebby/Documents/Notes`，清单按照已有公开文章的 `source_path` 建立。`source` 是相对于 Notes 根目录的路径，保留原大小写和中文文件名。

**首次同步前，先比较原笔记与网站文章。** 已有文章的标题、分类、固定网址等开头信息会保留，但正文会被本地笔记替换；网站导入时做过的正文修订也可能被覆盖。需要保留的修订应先合并回原笔记，或将该条设为 `"enabled": false`，继续直接编辑 `_posts`。

## 日常编辑与发布

在 Notes 中编辑完成后，进入网站仓库，先获取远程修改，再预览一篇笔记的同步结果：

```bash
cd /Users/sebby/Documents/The-Lyc.github.io
git pull --ff-only
.venv-notes-sync/bin/python bin/sync_notes.py --only 'MLSys/vLLM/源码阅读（3）：scheduler.md' --diff
```

默认是预览，**不会写入文件**；`--diff` 会打印正文差异。确认预览后，加 `--apply` 执行相同范围的同步：

```bash
.venv-notes-sync/bin/python bin/sync_notes.py --only 'MLSys/vLLM/源码阅读（3）：scheduler.md' --apply
git diff
git status
```

检查正文、图片路径及改动文件，确认都是准备公开的内容，再发布：

```bash
git add _posts assets/blog
git commit -m "Update blog notes"
git push origin main
```

当前网站在 `main` 收到提交后自动构建发布，可在 [GitHub Actions](https://github.com/The-Lyc/The-Lyc.github.io/actions) 查看结果。公开仓库中的文章、资源和历史提交都可以被访问，`published: false` 不能用来保存私人笔记。

不加 `--only` 会处理清单中全部启用的笔记；可以重复指定 `--only` 同步多篇。`--only` 必须与清单中的 `source` 完全一致。首次使用建议先同步一篇，避免一起覆盖大量正文。

```bash
# 全部启用文章：先预览，再写入
.venv-notes-sync/bin/python bin/sync_notes.py
.venv-notes-sync/bin/python bin/sync_notes.py --apply

# 临时更换笔记根目录或配置文件
.venv-notes-sync/bin/python bin/sync_notes.py --notes-root '/另一处/Notes' --config notes-sync.json
```

如果来源文件、引用资源缺失，或笔记链接指向未映射、停用同步、未公开的文章，脚本会在写入前报错。先修正链接、路径或补齐资源，再重新执行；也可以先用 `--only` 处理其他已经准备好的文章。所有选中文章通过验证后才会写入，避免同步一半才发现错误。

## 添加新文章

先在 Notes 中完成笔记和配图，再在 `notes-sync.json` 的 `posts` 数组中添加一项，例如：

```json
{
  "source": "MLSys/vLLM/我的新笔记.md",
  "target": "_posts/2026-10-03-vllm-new-note.md",
  "metadata": {
    "title": "我的 vLLM 新笔记",
    "date": "2026-10-03",
    "categories": ["MLSys"],
    "lang": "zh-CN"
  }
}
```

新文章必须提供 `title` 和 ISO 格式的 `date`，目标文件名使用 `YYYY-MM-DD-<slug>.md`。可选信息包括 `categories`、`tags`、`description`、`permalink`、`series`、`series_order` 和 `lang`。脚本默认添加 `layout: post`、`notes_import: true`、`source_path` 和按文件名生成的 `/blog/<slug>/` 固定网址。

原笔记自带的 YAML 开头信息不会覆盖网站文章信息；已有文章仍保留其原开头信息。新文章信息以清单的 `metadata` 为准。完成预览和同步后，除文章和资源外，记得提交新增的映射：

```bash
.venv-notes-sync/bin/python bin/sync_notes.py --only 'MLSys/vLLM/我的新笔记.md'
.venv-notes-sync/bin/python bin/sync_notes.py --only 'MLSys/vLLM/我的新笔记.md' --apply
git diff
git add notes-sync.json _posts assets/blog
git commit -m "Publish new vLLM note"
git push origin main
```

从清单移除笔记或设置 `"enabled": false` 只会停止同步，已经发布的文章会保留；取消公开需要另外修改网站仓库。

## 图片与笔记链接

笔记中的相对图片、附件路径以笔记所在目录为基准，引用资源会复制到该文章的 `assets/blog/<slug>/` 中，并改为兼容站点路径的链接。同目录资源保留相对目录结构；来自 Notes 其他目录的共享资源放在该文章资源目录的 `_shared/` 中。配套样式让图片随博客正文宽度缩放。建议使用相对路径，避免另一台电脑的绝对路径；Windows 绝对路径会报错，需先改成相对路径。

指向清单中其他已启用、已公开博客笔记的 `.md` 链接会转换为对应文章的固定网址。未映射的笔记不会跟随链接自动发布，需删除该链接、改为公开链接，或另行将准备公开的笔记加入清单。

脚本支持常用 Markdown 图片和链接、引用式链接、HTML 中带引号的 `src` / `href`，以及简单的 `![[image.png]]`、`[[note.md|label]]` 写法。代码块和行内代码不会改写；已有外部网址和网站 Liquid 链接会保留。同步前仍需检查图片是否齐全、链接是否指向准备公开的文章。
