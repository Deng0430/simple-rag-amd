# AUP 云端开发入门助手

一个使用官方资料回答 AUP 环境配置、程序运行、网页访问和文件保存问题的 RAG 网页。支持 AMD AUP / ROCm 云端推理、本地 Ollama 问答和 CPU 检索预览。回答附资料片段与来源，使用者自行核对和执行操作。

## 这是什么

- 默认 **X：BM25＋多语言 E5 等权 RRF 融合 → Top-3 基础片段 → 最多 512 token 的额外完整原文 → 模型回答**。
- 在“演示参数 → 证据方案”中可切换 H（融合 Top-3）、D（E5 Top-3）或“自定义检索演示”；自定义支持 TF-IDF、BM25、E5 与 Top-k 调整、检索对照。
- 支持资料筛选、原文及来源查看、运行记录和 CSV 下载。仅检索预览不调用生成模型。
- 来源清单包含 AUP 用户指南的 4 页正文和 16 页相关开发文档，另有本项目操作说明。其他工具仅导入清单指定页面，不抓取整个网站。

代码结构：

| 文件 | 用途 |
| --- | --- |
| `app.py`、`assets/style.css` | Streamlit 网页 |
| `knowledge.py`、`prepare_knowledge.py` | 下载、清洗、切分及管理资料 |
| `retrievers.py`、`embedding_model.py` | TF-IDF、BM25、E5 与向量缓存 |
| `hybrid_retrieval.py`、`hybrid_runtime.py` | D/H/X 检索、证据预算与问答流程 |
| `answer_protocol.py`、`structured_model.py` | 证据格式、JSON 校验及本地结构化生成 |
| `rag.py` | 生成后端、单检索器流程和 CSV 导出 |
| `data/` | 公开来源清单、6 个快捷问题、项目操作说明及来源声明 |

## 怎么装

### AMD AUP 云端

选择预装 ROCm PyTorch 的 **VSCode Server GPU**。在终端执行：

```bash
cd /home/jovyan
git clone https://github.com/Deng0430/simple-rag-amd.git
cd simple-rag-amd
bash start_aup.sh
```

脚本检查 GPU、创建继承平台 PyTorch 的虚拟环境、安装依赖、导入资料并启动网页。缺少 `ensurepip` 时会使用 PyPA 官方 virtualenv。不要用普通 PyPI 的 PyTorch 覆盖平台的 ROCm 构建。

### Windows 本地

需要 Python 3.12 和网络；模型问答还需要 [Ollama](https://ollama.com/download)。在 PowerShell 执行：

```powershell
git clone https://github.com/Deng0430/simple-rag-amd.git
Set-Location simple-rag-amd
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_knowledge.py
ollama pull qwen2.5:3b
ollama list
```

只做检索预览时可以跳过 Ollama 安装和模型下载。权重、下载的官方正文和向量索引保存在本机，不随 Git 仓库分发；模型和资料首次准备所需时间取决于网络及硬件。

## 怎么跑

### 云端启动、更新和访问

已安装时，在 AUP 终端运行 `bash /home/jovyan/simple-rag-amd/start_aup.sh`。也可用 VS Code 打开项目目录，按 **Ctrl+Shift+B** 运行启动任务。

启动后保留终端，在 Code Server 的 **PORTS 面板打开 8501 的转发地址**。云端终端显示的 `localhost:8501` 不会自动指向你的本机浏览器。

更新时先在旧服务终端按 Ctrl+C，再执行：

```bash
cd /home/jovyan/simple-rag-amd && git pull --ff-only && bash start_aup.sh
```

如果 Git 报本地修改或分支分歧，先检查 `git status` 并保存本地修改；拉取失败时不会继续启动。

### 本地启动

在项目目录运行：

```powershell
.\start_local.ps1
```

脚本检查模型和依赖，在需要时后台启动 Ollama，然后打开服务 `http://127.0.0.1:8501/`。它不自动安装依赖或下载模型。模型使用自定义目录时，可加 `-ModelsPath '实际模型目录'`；该参数只对脚本新启动的 Ollama 服务生效。端口占用时可加 `-Port 8502`。

仅检索预览也可直接启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1
```

### 使用示例

1. 选择“模型问答”，保留 X，输入“为什么终端提示 python: command not found？”；核对回答中的命令与来源。
2. 选择“仅检索预览”，输入“Streamlit 启动了，为什么网页打不开？”；展开排名靠前的原文，查看代理和端口说明。
3. 在“证据方案”选择“自定义检索演示”，对同一问题比较三种检索器；分数只用于各自的排序，不能直接跨算法比较大小。

每次运行后可以下载 CSV。刷新网页、重启进程或停止云资源前，先将需要的结果保存到本机。

CSV 中每个结果占一行，保留问题、回答、模型、方案、token、耗时、片段 ID 和分数等已有字段；D/H/X 模型问答也保留原始模型输出（`raw_answer`）。另附两列证据快照：

- `evidence_text`：可直接阅读的证据，按 `[P1]`、`[P2]` 顺序包含片段标题、ID、分数、适用范围、完整原文，以及来源标题、链接、类型和核对日期（有记录时）。
- `passages_json`：同一批证据的完整 JSON 数组，保留原有片段字段、所有来源和检索元数据；每段的 `citation_id` 与网页引用编号对应，方便后续程序分析。

证据取自本次结果，下载时不重新检索，因此后续资料更新也不会改变已保存的证据。无证据时两列分别为空和 `[]`；仅检索预览的答案和模型 token 仍为空。Excel 中可对 `evidence_text` 启用“自动换行”查看。用于失败分析时保留原始 CSV，再另行记录判断。

## 效果如何

网页展示检索片段、来源链接、模型回答、耗时和实际输入/输出 token；X 同时展示基础片段及额外片段数量和补充预算。格式失败、空短答案、拒答与缺失或无效引用会给出相应提示。引用编号用于核查，并不证明所有输出都正确。

本地 D/H/X 使用固定的 Qwen2.5-3B / Q4_K_M 模型版本、8192 上下文和 512 输出上限，使用实际 tokenizer 检查预算。云端默认采用 `Qwen/Qwen3-4B-Instruct-2507`，使用该模型自己的模板和 tokenizer。本地与云端模型不同，不能直接套用另一环境的质量或性能结论。

运行随仓库提供的系统单元测试：

```bash
python -m unittest discover -s tests -v
```

测试使用小型合成输入，覆盖检索融合、证据预算、资料解析、后端响应、错误处理和导出；不需要下载生成模型。功能检查不能代替对真实答案的人工核对。

## 已知问题

- 系统不能查看账户、配额、文件和服务器状态，也不会替使用者执行命令。回答可能混淆参数、遗漏条件或不恰当拒答。
- 首次使用会下载资料、嵌入模型及所需生成模型；模型加载和资料索引可能较慢。默认复用缓存，更新资料使用 `python prepare_knowledge.py --refresh`。
- E5 的相似度不是相关性概率，当前没有针对所有无关问题校准拒答阈值；尚未自动逐句验证引用是否支持答案。
- 本地 D/H/X 为隔离结构化生成的服务状态，每次请求前重载同一模型，耗时包含重载。需要其他本地模型或上下文窗口时，使用自定义检索入口。
- 模型输出可能达到长度上限。请查看提示和原文，不直接执行不完整的建议。
- AUP 会话到期后需重新启动服务；代码推送不会备份被忽略的缓存和 CSV。重要结果应另行下载保存。

公开仓库仅包含系统部署代码与必要运行资料。个人报告、实验题库、评分和运行记录均不随当前版本发布。代码许可见 [LICENSE](LICENSE)，第三方资料和模型归属见 [data/ATTRIBUTION.md](data/ATTRIBUTION.md)。
