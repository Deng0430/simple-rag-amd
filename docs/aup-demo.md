# AUP 入门助手：部署与使用

这个网页从固定的中文指南中检索 AUP 云端开发相关说明，生成简短回答并展示引用。用户也可以直接浏览指南，或在本地预览检索结果。

## 适用范围

知识库包含 8 条指南和 6 道示例题，涉及环境选择、Python 命令、虚拟环境、GPU、网页启动、端口转发、文件保存和会话管理。它不查询账户余额或服务器状态，不会替用户执行命令、恢复文件或控制会话。账号状态、资源配额和可用硬件请以 AUP 页面为准。

资料按来源标注适用范围；特定环境中的观察不能代表所有 AUP 镜像。模型只会收到检索出的指南。若回答没有得到资料支持，应以来源原文及平台说明为准。

## 部署步骤

1. 在 AUP 启动 **Development Environments → VSCode Server GPU**，打开 Code Server。
2. 将仓库克隆到持久目录 `/home/jovyan`。若仓库已存在，可运行 `cd /home/jovyan/simple-rag-amd && git pull --ff-only` 更新。
3. 在 VS Code 使用 **File → Open Folder** 打开项目文件夹。按 **Ctrl+Shift+B** 启动默认任务；也可在项目终端运行 `bash start_aup.sh`。
4. 服务启动后，在 **PORTS 面板**打开 8501 端口的转发地址。
5. 首次点击问答按钮时会下载或加载模型。网页进程可在任务终端按 Ctrl+C 停止；关闭 AUP 资源请使用 Hub 上的 **Stop my server**。

仅在终端切换目录不会加载项目的 VS Code 任务。任务使用 `start_aup.sh` 检查 GPU、准备环境、安装依赖并启动网页；它不会启动 AUP 资源或替用户更新仓库。若快捷键被浏览器占用，可在 VS Code 命令面板选择 **Tasks: Run Task → 启动 AUP 入门助手**。

脚本优先使用 `python3`，在标准虚拟环境不可用时会使用 PyPA 官方 virtualenv zipapp，并复用平台 PyTorch。请勿安装普通 PyPI PyTorch 覆盖 ROCm 构建。

## 本地检索预览

安装 Streamlit 和 scikit-learn 后，运行 `python -m streamlit run app.py`，打开页面并选择“仅检索预览”。预览模式不需要模型或 GPU，也不会生成答案。

## 资料来源

| 来源 | 支持的内容 |
| --- | --- |
| [AUP Platform Basics](https://amdresearch.github.io/aup-learning-cloud/user-guide/platform-basics.html) | 资源选择、持久目录及会话管理 |
| [AUP Code Server Guide](https://amdresearch.github.io/aup-learning-cloud/user-guide/code-server-guide.html) | 终端、PORTS 面板和网页访问 |
| [Python venv](https://docs.python.org/3.12/library/venv.html) | 虚拟环境及系统包访问 |
| [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html) | ROCm/HIP 下的 `torch.cuda` 接口 |
| [Streamlit run](https://docs.streamlit.io/develop/api-reference/cli/run) 与 [config.toml](https://docs.streamlit.io/develop/api-reference/configuration/config.toml) | 网页运行及服务配置 |
| [项目启动脚本](../start_aup.sh) | 环境检查、依赖准备与网页启动 |
| [资料署名与来源](../data/ATTRIBUTION.md) | 指南和示例数据的来源、许可 |

## 验证与边界

网页的本地验证包括示例问答、指南来源展开、检索预览和无匹配提示。GPU 问答需要在具有可用 ROCm PyTorch 的 AUP 资源上运行。评测脚本逐题输出 CSV 和环境快照至本地 `runs/`，不需要把生成结果加入 Git 仓库。

检索基于字符 TF-IDF，相似度是词面相关性指标，不是答案正确概率。无匹配时网页会跳过模型生成；存在部分词语重合但资料仍不足时，模型仍可能生成不准确内容。引用材料供用户核对，不能替代平台官方说明。
