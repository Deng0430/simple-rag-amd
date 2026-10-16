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

TF-IDF、BM25 和 E5 稠密检索均在 CPU 运行。生成操作建议使用 AUP GPU 上的 Qwen/Qwen3-4B-Instruct-2507。没有 GPU 时选择“仅检索预览”。首次选择稠密检索需要下载多语言嵌入模型并为文档编码，后续使用磁盘向量缓存；这与生成模型的下载和 GPU 加载是两个阶段。

依赖已安装时，在项目目录启动 CPU 网页：

```bash
python3 -m streamlit run app.py
```

`start_aup.sh` 专用于 AUP GPU 环境，会检查 ROCm 和 GPU，不用于启动 CPU 预览。若 8501 已被自己的旧进程占用，先在那个进程的终端按 Ctrl+C，再重新启动。

## 完整资料库与检索对照

正式知识库由 data/source_manifest.json 中的 AUP 官方用户指南和相关开发文档原文，加本项目这份使用说明组成。每个片段记录原文页面、章节和抓取日期。旧的 40 条中文指南只保留作历史测试数据，不重复加入检索。

准备官方文档：

```bash
python3 prepare_knowledge.py
```

下载正文和索引保存在项目 `.cache`，不自动上传 GitHub。默认复用已下载文档；更新需显式运行 `python3 prepare_knowledge.py --refresh`。导入脚本只读取清单中的公开页面，不跟随正文中的指令执行操作。

在“演示参数”切换 TF-IDF、BM25 或 E5。可对同一问题并排比较三种检索器。该对比只检索，不调用生成模型；分数属于各算法自己的尺度，不能直接跨算法比较数值大小。E5 默认保留 Top-k，不把未校准相似度当作相关性概率。相关资料是否足够应结合原文判断。

## 模型运行与计时

首次模型问答会下载（如未缓存）、加载并预热生成模型；模型资源在网页进程中复用。检索耗时包含问题编码与搜索，不含首次模型下载和索引构建。运行记录单独记录检索器准备耗时。跑命令行模型评测前先停网页，避免同时加载两份生成模型。切换到检索预览不保证卸载已经在进程中缓存的生成模型。

## 保存结果和更新代码

每次问答或检索后可点击“下载本次结果 CSV”。网页结果仅保存在当前连接会话中，刷新或重启前先下载。CSV 记录检索器、Top-k、资料 ID、分数、耗时及实际产生的模型 token。仅检索的答案和模型 token 为空。

本项目 .gitignore 忽略 runs、results、.cache、模型权重和虚拟环境。代码推送到 GitHub 不代表这些结果也已备份；重要文件应另行下载或保存在 /home/jovyan。更新代码前停止旧网页，然后运行：

```bash
cd /home/jovyan/simple-rag-amd && git pull --ff-only && bash start_aup.sh
```

若拉取因本地修改或网络失败，&& 后面的启动不会执行。先查看 git status 和 git diff、保留本地修改，再处理更新。
