# Notes 技术笔记迁移与维护

本次从 `Notes.zip` 选取系统架构、MLSys、模型笔记、论文阅读和课程笔记，导入 36 篇文章：32 篇可公开展示，4 篇因缺少配图暂存为 `published: false` 的技术草稿。文章引用的 114 个资源文件约 18 MB，按篇保存在 `assets/blog/<slug>/`。

原网站的 33 篇模板示例文章保留源文件并设为 `published: false`；示例外部博客源已停用，博客列表展示本次导入的公开文章。

迁移时已重写本地和 Windows 图片路径，让图片随页面宽度缩放，并适配公式与提示块。Linux 进程笔记中五张无法直接加载的外部配图，保留说明并改为原文链接；已随包提供的图片和证据附件保留原文件。

个人内容、计划、测试流水账、空白文件和未成文草稿没有复制到网站仓库，原始材料继续保留在原 `Notes.zip` 中。`paper/CopyEngine/survey.md` 也暂未导入：它的引用仍是无法解析的内部来源标记，需要补全真实的论文或资料链接后再整理发布。原始压缩包和完整笔记库不应上传到公开网站仓库。

## 文章组织

导入清单为 [`_data/blog_import.json`](../_data/blog_import.json)，记录原路径、文章文件、固定地址及资源。文章开头的 `source_path` 保留原笔记相对路径，便于回查来源；`notes_import: true` 将文章纳入[技术笔记与专题页](../_pages/blog-topics.md)。

| 原笔记主题                  | 分类键         | 专题页名称   |
| --------------------------- | -------------- | ------------ |
| 系统架构、Linux、实时系统   | `Architecture` | 系统架构     |
| CUDA、推理框架、Agent 系统  | `MLSys`        | 机器学习系统 |
| Transformer、Decoder-only   | `Models`       | 模型笔记     |
| 论文阅读与 CPU–GPU I/O 报告 | `Papers`       | 论文阅读     |
| CSE234                      | `Courses`      | 课程笔记     |

原笔记没有一致、可信的原发表日期，因此文章日期统一设为迁移日 **2026-09-30**，作为首次公开日期，不从图片名称推断写作日期。四篇草稿补齐后，应使用各自实际首次公开的日期。

文件名使用 `YYYY-MM-DD-<slug>.md`，中文标题保留在 `title` 中。每篇显式设置 `/blog/<slug>/` 形式的 `permalink`；以后修改标题、日期或分类时，应保留这个地址，避免已有链接失效。

专题页的数据在 [`_data/blog_topics.yml`](../_data/blog_topics.yml)。页面按分类列出普通文章，按 `series` 和 `series_order` 排列系列文章，不重复列出系列章节。目前包含：

- vLLM 源码阅读：第 2、3、4 篇，保留原章节编号；第 1 篇原文是占位笔记，尚未导入。
- llama.cpp 源码阅读：第 1、2、3 篇。
- CSE234：深度学习系统基础在前，并行计算与分布式训练在后。

`series` 和 `series_order` 是本专题页使用的自定义文章信息；它们不会自动在文章页面生成上一篇、下一篇或系列目录。

## 以后在哪里编辑

这是一次性导入。**已公开文章以网站仓库中的 `_posts` 版本为准**；本地 Notes 仓库继续保存私人笔记、草稿和原始材料。不要用原 Notes 文件再次覆盖已经在网站仓库修改过的文章。

本地可用 Typora 打开网站仓库中的文章；云端可在 GitHub 网页编辑同一个 `_posts` 文件。两种方式都修改正文和开头的文章信息，图片及下载文件则放在该篇对应的 `assets/blog/<slug>/` 目录中。提交前检查标题、描述、分类、标签和图片，保留 `permalink`、`source_path` 与 `notes_import`；系列文章也保留正确的 `series` 和 `series_order`。

网站中的资源链接使用 `relative_url`，例如：

```markdown
![示意图]({{ '/assets/blog/vllm-kv-cache-management/diagram.png' | relative_url }}){: .img-fluid loading="lazy" }
```

Typora 可能无法在本地预览这类 Liquid 链接，最终效果以网站预览为准。上传新图片时不要保留另一台电脑的绝对路径。复制目录中的旧文章链接时，改为对应文章的固定 `permalink`；同时需要兼容站点前缀时，使用相同的 `relative_url` 写法。

当前[发布流程](../.github/workflows/deploy.yml)在 `main` 或 `master` 收到提交时自动构建和发布，也可以在 GitHub Actions 手动运行 `Deploy site`。网页编辑提交到这些分支后，等待该流程成功再检查网站。

## 四篇待补图的技术草稿

这些文件已保存到网站仓库，当前不会进入正常网站构建或专题页。`published: false` 只控制网页展示，文件内容仍可在公开 GitHub 仓库中阅读。

| 文章文件                                                                           | 原包缺少的资源                                                                                                             |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [GeDES](../_posts/2026-09-30-paper-gedes.md)                                       | `gedes_dag_full_diagram.svg`                                                                                               |
| [HeteroInfer](../_posts/2026-09-30-paper-heteroinfer.md)                           | `image-20260410002513991.png`、`image-20260409234232369.png`、`image-20260409234926717.png`、`image-20260410001546466.png` |
| [GPU 统一内存与资源共享](../_posts/2026-09-30-paper-gpu-unified-memory-sharing.md) | `image-20250109161420212.png`                                                                                              |
| [Attention Residuals](../_posts/2026-09-30-paper-attention-residuals.md)           | `image-20260327143435545.png`                                                                                              |

找回资源后，把文件放到清单对应的 `assets/blog/<slug>/` 目录，将正文中的“图片待补充”说明替换为实际图片链接，并检查显示效果。随后删除文章开头的 `published: false`，或改为 `published: true`，并同步更新清单中的 `published`、`assets` 和 `missing_assets`。

## 后续页面改版

文章内容、专题数据和分类可以继续在此仓库维护。页面运行时、样式和功能的修改按现有[架构说明](ARCHITECTURE.md)与[所有权边界](BOUNDARIES.md)处理。
