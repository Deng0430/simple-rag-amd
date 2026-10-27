# 本项目使用说明

这份说明描述 simple-rag-amd 自己的运行方式，与平台官方文档一起检索。旧版 40 条通用中文 FAQ 不参与正式检索。

## 启动本项目网页

本项目云端默认目录为 `/home/jovyan/simple-rag-amd`。进入 AUP 的 GPU Code Server 后，选择 Terminal → New Terminal，执行：

```bash
bash /home/jovyan/simple-rag-amd/start_aup.sh
```

脚本会切换到自己的目录，检查 ROCm PyTorch、准备项目环境并在 0.0.0.0:8501 启动 Streamlit。保留启动终端中的进程，在 Code Server 的 PORTS 面板打开 8501 的转发地址。进入 VS Code 并不等于项目程序已经启动。本机浏览器的 localhost 不会自动指向云端。按 Ctrl+C 可停止网页；停止整个 AUP 资源需要到 Hub 点击 Stop my server。

## Python 命令与虚拟环境

本项目实测的 AUP 镜像使用 `/usr/bin/python3`，`python: command not found` 时先运行 `python3 --version`。项目环境创建后使用 `.venv/bin/python`，不要混用不同解释器安装与运行依赖。

若创建环境提示 ensurepip 不可用，使用本项目 `start_aup.sh`。它先尝试标准 venv，失败后下载 PyPA 官方 virtualenv zipapp，创建继承系统包的环境，不需要 sudo。脚本生成 `.cache/aup-constraints.txt`，约束 torch 为平台构建，再安装 requirements.txt。不要用普通 PyPI 的 torch 覆盖平台预装的 ROCm PyTorch。AMD 下 `torch.cuda` 是 ROCm 后端复用的接口。

## 本地或 CPU 检索演示

TF-IDF、BM25 和 E5 稠密检索均在 CPU 运行。AUP 默认用 Transformers 和 Qwen/Qwen3-4B-Instruct-2507 生成操作建议；本地也可在“模型与运行环境”中选择 Ollama，使用已经安装的 qwen2.5:3b。Ollama 独立于网页 Python 环境运行，由服务决定 CPU/GPU 分配；网页未检测到 PyTorch GPU 不代表 Ollama 不能问答。生成服务不可用时仍可选择“仅检索预览”。首次 E5 检索的嵌入模型下载和编码，与生成模型的加载是两个阶段。

依赖已安装时，在项目目录启动 CPU 网页：

```bash
python3 -m streamlit run app.py
```

`start_aup.sh` 专用于 AUP GPU 环境，会检查 ROCm 和 GPU，不用于启动 CPU 预览。若 8501 已被自己的旧进程占用，先在那个进程的终端按 Ctrl+C，再重新启动。

Windows 本地已安装 Ollama、项目依赖和模型时，可在项目目录的 PowerShell 运行 `./start_local.ps1`。模型保存在自定义目录时，用 `-ModelsPath` 指定实际目录；该参数仅在脚本新启动 Ollama 服务时生效。脚本检查已安装模型，不自动下载权重；已有 Ollama 服务会直接复用。

## 完整资料库与检索对照

正式知识库由 data/source_manifest.json 中的 AUP 官方用户指南和相关开发文档原文，加本项目这份使用说明组成。每个片段记录原文页面、章节和抓取日期。旧的 40 条中文指南只保留作历史测试数据，不重复加入检索。

准备官方文档：

```bash
python3 prepare_knowledge.py
```

下载正文和索引保存在项目 `.cache`，不自动上传 GitHub。默认复用已下载文档；更新需显式运行 `python3 prepare_knowledge.py --refresh`。导入脚本只读取清单中的公开页面，不跟随正文中的指令执行操作。

在“演示参数”切换 TF-IDF、BM25 或 E5。可对同一问题并排比较三种检索器。该对比只检索，不调用生成模型；分数属于各算法自己的尺度，不能直接跨算法比较数值大小。E5 默认保留 Top-k，不把未校准相似度当作相关性概率。相关资料是否足够应结合原文判断。

## 模型运行与计时

Transformers 首次问答会下载（如未缓存）、加载并预热模型，模型资源在网页进程中复用。Ollama 复用独立服务中的已安装模型，记录实际 token 和加载、提示处理、生成耗时；没有测量的 PyTorch 显存字段留空。检索耗时包含问题编码与搜索，不含首次下载和索引构建，索引准备耗时单独记录。批量评测前避免网页同时问答；Transformers 还需避免加载两份模型。切换到检索预览不保证卸载已缓存的模型。本地与 AUP 的模型、量化、设备不同，结果分别保存和解释。

## 保存结果和更新代码

每次问答或检索后可点击“下载本次结果 CSV”。网页结果仅保存在当前连接会话中，刷新或重启前先下载。CSV 记录检索器、Top-k、资料 ID、分数、耗时及实际产生的模型 token。仅检索的答案和模型 token 为空。

本项目 .gitignore 忽略 runs、results、.cache、模型权重和虚拟环境。代码推送到 GitHub 不代表这些结果也已备份；重要文件应另行下载或保存在 /home/jovyan。更新代码前停止旧网页，然后运行：

```bash
cd /home/jovyan/simple-rag-amd && git pull --ff-only && bash start_aup.sh
```

若拉取因本地修改或网络失败，&& 后面的启动不会执行。先查看 git status 和 git diff、保留本地修改，再处理更新。
