# AUP 云端开发入门助手

一个运行在 **AMD AUP Learning Cloud** 上的资料问答网页。学生描述环境配置、网页访问或文件保存问题，系统检索官方文档，再用 GPU 模型整理带引用的操作建议。

## 功能与资料

- 导入 AUP 用户指南全部 4 页正文，加 Python、pip、Git、VS Code、Streamlit、PyTorch HIP、Hugging Face Hub 的指定开发文档，共 20 页，另附本项目使用说明。
- 在“演示参数”切换 **TF-IDF、BM25、多语言 E5 稠密检索**，调整 Top-k，或并排比较三种检索器的排名、原文、分数和查询耗时。
- 支持仅检索预览、GPU 模型问答、来源展开和 CSV 下载。资料库支持关键词、主题筛选与分页。
- 旧版 40 条原创 FAQ 移至 `data/archive/aup_faq_v2.json`，仅供历史测试；它们的正文不加入当前检索库。

“完整”指 [资料清单](data/source_manifest.json) 中的 AUP 用户指南 4 页正文；其他工具导入的是清单中的具体页面，未抓取所有网站。切分数量随上游正文变化，页面显示实际导入状态。

这是课程 demo，不能查询账户或服务器状态，也不会替用户执行命令。回答依据和资源状态请结合原文、实际版本与 AUP 页面核对。详见[部署与演示说明](docs/aup-demo.md)。

## 在 AUP 启动

在 AUP 选择 **Development Environments → VSCode Server GPU**。将仓库克隆至 `/home/jovyan`，进入项目后执行：

```bash
bash start_aup.sh
```

已经克隆时，先停止旧网页，再更新并启动：

```bash
cd /home/jovyan/simple-rag-amd && git pull --ff-only && bash start_aup.sh
```

也可用 **File → Open Folder** 打开项目文件夹，按 **Ctrl+Shift+B** 运行默认任务。服务监听 8501，从 Code Server **PORTS 面板**打开转发地址；云端 localhost 不会自动指向本机浏览器。

脚本会检查 GPU、准备环境、安装依赖并首次导入官方文档，复用平台已有的 ROCm PyTorch。缺少 ensurepip 时自动使用 PyPA virtualenv。首次模型问答会加载 `Qwen/Qwen3-4B-Instruct-2507`；首次 E5 检索会下载嵌入模型并建立向量索引。提前运行一次可减少现场等待。**不要用普通 PyPI PyTorch 覆盖平台的 ROCm 构建。**

## 本地运行与索引

在 Python 环境中执行（AUP 使用 `.venv/bin/python`，Windows 可使用 `.venv\Scripts\python`）：

```bash
python -m pip install -r requirements.txt
python prepare_knowledge.py
python -m streamlit run app.py
```

没有 GPU 时选择“仅检索预览”，可以运行三种检索器；此时不生成答案，模型 token 留空。文档未导入时仅有项目说明，网页显示提示并提供准备按钮。

默认复用文档快照；重新抓取使用 `python prepare_knowledge.py --refresh`。正文按章节和段落切分，保留命令、出处与抓取日期。

| 检索器 | 索引与查询方式 |
| --- | --- |
| TF-IDF | scikit-learn 字符 2—4 gram 稀疏向量，按余弦相似度排序 |
| BM25 | jieba 中文分词及英文 API 词元，使用 BM25Okapi 排序 |
| E5 | intfloat/multilingual-e5-small，384 维，CPU ONNX int8；对归一化文档向量做余弦检索 |

当前规模使用 **NumPy 向量矩阵与本地缓存文件**，无需另开向量数据库服务。文档向量按正文和嵌入模型版本生成缓存键，资料变化后自动重建。E5 使用 query: / passage: 前缀；ONNX 文件来自 Xenova/multilingual-e5-small，下载时记录实际 revision 并检查文件完整性。

文档快照在 `.cache/knowledge/`，嵌入模型与 `.npz` 向量索引在 `.cache/retrieval/`；均不提交 GitHub。云端和其他电脑首次运行时各自准备缓存。

## 对比与验证

运行 6 道快捷问题，导出三种检索器的真实结果：

```bash
python evaluate.py --preview --retriever all --limit 6 --output runs/retrieval_comparison.csv
python -m unittest discover -s tests -v
```

GPU 问答和同一检索器的 Top-k 对比：

```bash
python evaluate.py --retriever bm25 --limit 6 --output runs/aup_demo.csv
python evaluate.py --retriever dense --top-k 3 6 --limit 6 --output runs/dense_demo.csv
python summarize_results.py runs/aup_demo.csv
```

批量 GPU 推理前先停止网页，避免加载两份生成模型。CSV 记录检索器、资料 ID、分数、耗时及实际生成 token。汇总脚本仅处理单一检索器的模型回答，正确率来自另行人工标注；它不接受仅检索结果。

三种分数尺度不同，**不能横向比较数值大小，也不是答案正确率**。E5 默认保留 Top-k，尚未校准无关问题的拒答门槛。查询耗时包含问题编码与搜索，首次下载、索引准备和生成模型加载另算。旧 FAQ 与 SQuAD 的 GPU 记录不代表新资料库已经完成 GPU 复测。

历史数据可显式指定：

```bash
python evaluate.py --data data/archive/aup_faq_v2.json --preview --limit 40 --output runs/legacy_preview.csv
python evaluate.py --data data/squad_demo.json --top-k 3 6 --preview --output runs/squad_preview.csv
```

## 文件与许可

| 文件 | 用途 |
| --- | --- |
| app.py、assets/style.css | 网页及样式 |
| rag.py、retrievers.py、embedding_model.py | 问答、三种检索器、嵌入模型准备 |
| knowledge.py、prepare_knowledge.py、data/source_manifest.json | 文档导入、章节切分及来源清单 |
| docs/project-usage.md | 参与检索的项目自身操作说明 |
| evaluate.py、summarize_results.py | 结果导出及模型回答汇总 |
| start_aup.sh、.vscode/tasks.json | AUP 快捷启动 |
| data/archive/、data/squad_demo.json | 历史测试资料 |

代码和原创项目说明采用 MIT。第三方文档、SQuAD 数据、模型及商标保留各自许可；下载的原文与模型权重不随仓库再分发。来源与许可见 [ATTRIBUTION.md](data/ATTRIBUTION.md)。
