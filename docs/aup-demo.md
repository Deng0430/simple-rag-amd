# AUP 入门助手：部署与演示

这个小型 RAG 网页面向初次在 AUP 开发的学生。它从官方文档和项目说明中查找依据，再生成操作建议；也能直接查阅原文、比较检索效果。

## 资料范围

当前知识库为 `aup_documents_v1`：AUP 用户指南全部 4 页正文与 16 页相关开发文档，加 `docs/project-usage.md`。清单见 `data/source_manifest.json`。网页显示实际成功导入的页面与片段数量，不把未下载的页面计入。这里没有收集 AUP 网站之外的全部工具文档。

第三方正文下载至 `.cache/knowledge/`，按章节和约 1000 字符切分，记录原始页面、章节锚点与抓取日期。本项目说明保留自身的启动方法、ROCm 约束和部署经验。旧版 40 条中文 FAQ 已归档，正文不参与正式检索；首页沿用其中 6 道问题的提问文字。

## 启动步骤

1. 在 AUP 启动 **Development Environments → VSCode Server GPU**，打开 Code Server。
2. 用 **File → Open Folder** 打开 `/home/jovyan/simple-rag-amd`。已克隆时先停止旧网页，按需执行 `git pull --ff-only` 更新。
3. 按 **Ctrl+Shift+B** 启动默认任务，或在终端执行 `bash /home/jovyan/simple-rag-amd/start_aup.sh`。
4. 服务启动后，从 Code Server **PORTS 面板**打开 8501 的转发地址。
5. 首次使用 E5 或生成模型需要下载和准备。演示前先各运行一次；按 Ctrl+C 停止网页，在 Hub 使用 **Stop my server** 停止整个资源。

仅在终端进入目录不会自动加载 VS Code 任务；快捷键失效时，可用命令面板 **Tasks: Run Task → 启动 AUP 入门助手**。启动脚本检查 GPU，准备继承系统包的虚拟环境，安装依赖并首次导入原文，保留平台 ROCm PyTorch。它不会替用户启动 AUP 资源或更新 Git 仓库。

本地预览可运行：

```bash
python -m pip install -r requirements.txt
python prepare_knowledge.py
python -m streamlit run app.py
```

没有 GPU 时选择“仅检索预览”。下载未完成时，页面显示当前资料状态，可以点击“准备官方文档库”重试。已有文档默认不重新下载；显式更新用 `python prepare_knowledge.py --refresh`。

## 建议演示流程

1. 选择“仅检索预览”，展开“演示参数”。保持 Top-k 为 3，选择“对比三种检索器（只检索）”。
2. 输入 `为什么 AMD 显卡也使用 torch.cuda 接口？`，点击“比较检索效果”。查看三个检索器的排名、资料内容、来源链接及耗时。
3. 选择“单次检索”和一种检索器，切换“模型问答”，询问本项目的启动方法。核对回答中的引用与命令，再下载 CSV。
4. 打开“资料库”，按主题或关键词查找一段原文，展示完整来源。无需逐条演示全部资料。

TF-IDF 使用字符 n-gram，BM25 使用中文分词和 API 词元，E5 使用多语言嵌入。稠密检索在 CPU 编码与搜索，生成模型在 AMD GPU 运行。这两个模型有独立的缓存和准备阶段。

当前片段数量较少，E5 文档向量保存在 `.cache/retrieval/*.npz`，查询时编码问题并与矩阵计算余弦相似度，无需运行额外的数据库。正文或嵌入模型版本变化时自动重新编码。缓存不随 GitHub 发布，云端首次使用会自行准备。

## 对比与结果记录

```bash
python evaluate.py --preview --retriever all --limit 6 --output runs/retrieval_comparison.csv
python evaluate.py --retriever dense --limit 6 --output runs/aup_demo.csv
```

AUP 使用 `.venv/bin/python`。批量生成前停网页，避免同时加载两份生成模型。结果和环境快照保存在本地 `runs/`，不自动上传 GitHub；网页 CSV 应在刷新或重启前下载。

三种检索器对比只检索，答案、生成模型与 token 字段为空。分数不是正确率，也不能跨算法比较数值。检索计时包括问题向量化与搜索，不包括文档索引首次构建；索引准备时间单独记录。

旧 FAQ/SQuAD 结果仅对应当时的资料版本。新原文库的本地检索测试不等于完成真实 AMD GPU 问答验证。报告数据应使用当前版本重新生成并人工核对答案。

## 使用边界

助手不读取账户配额、服务器状态或本地文件，不代替用户执行命令。官方文档中的通用例子未必适用于本项目，应核对版本和项目前提，尤其不要用普通 torch 安装覆盖 ROCm 构建。E5 的语义门槛尚未校准，无关问题也可能返回片段；引用供核对，不能保证回答正确。

资料的完整 URL 与许可说明见 [资料清单](../data/source_manifest.json) 和 [ATTRIBUTION.md](../data/ATTRIBUTION.md)。
