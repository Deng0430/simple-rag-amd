# 模块、功能与验证范围

更新日期：2026-11-16。功能代码基线：`ffda90f9fac966379e76dfd975cff33140ab8281`；本轮只整理 README、架构和验证文档，未改变功能代码。下列说明按实际材料记录，不将本地维护者检查当作独立使用者验证。

## 文档与评测入口

| 内容 | 可定位材料 | 当前范围 |
| --- | --- | --- |
| 安装、运行与示例 | [README](../README.md)、[AUP 部署说明](aup-demo.md) | 五节使用说明，CPU 预览、本地 Ollama、AUP 路径及至少三个示例 |
| 架构与模块 | [架构图和说明](system-architecture.md) | 可编辑 SVG，区分网页与实验入口 |
| 三指标与基线 | [公开结果](../experiments/README.md)、[第一轮](condition-experiment-results.md)、[第二轮](retrieval-improvement-v2.md) | EM、Token-F1、AI 辅助 Slot-F1；A/D 检索基线与对照，含失败分析 |
| 统计复算 | `python experiments/verify_results.py` | 156 个逻辑位置、九行主表；不重新生成或标注 |
| 单元测试 | `python -m unittest discover -s tests -v` | 现有 60 项测试通过；覆盖具体逻辑，不能代表所有网页交互 |
| 本地模型接入 | README 的 Ollama 命令行示例 | 维护者已实际检索、生成、保存 CSV，已有缓存，不是全新电脑安装测试 |
| 独立使用者验证 | 待真实试用后补充匿名记录 | 尚未完成，不能据此声称陌生环境 30 分钟内跑通 |

## 代码规模口径

在上述代码基线上，根目录 16 个 Python 源文件合计约 **2542 有效物理行**。统计使用 Python AST 识别模块、类和函数的 docstring，再统计 tokenize 中非注释、非空白 token 所在行，并排除 docstring 行。未计测试、Markdown、JSON、CSS、启动脚本或第三方依赖。

这是明确口径下的规模自查，不同工具的“有效行”定义可能不同。规模不能代替功能质量；项目已经有完整的导入、检索、生成、核查和导出链条，无需为增加数量而添加空壳功能。至少资料、检索、生成、网页、批量检查、实验这六组职责可以独立定位，具体文件见 [架构模块表](system-architecture.md)。

仅网页、资料及问答链条的 `app.py`、`rag.py`、`retrievers.py`、`embedding_model.py`、`knowledge.py`、`prepare_knowledge.py` 六个文件约997有效行，不计实验和测试仍超过800行。当前功能属于固定RAG流程，Agentic方向是后续研究，详见 [技术决策与评测边界](technical-decisions.md)。

## 已实现功能与检查边界

| 功能 | 实现位置 | 验证或限制 |
| --- | --- | --- |
| 清单式官方资料导入及缓存 | `prepare_knowledge.py`、`knowledge.py` | 本机现有 20 页资料与项目说明；上游变化需重新记录 |
| 章节、段落及代码边界切分 | `knowledge.chunk_section` | 有单元测试；不代表任意网页版式都能无损导入 |
| 三种检索器及向量缓存 | `retrievers.py`、`embedding_model.py` | 已检查三检索器批量示例；E5 门槛未校准 |
| 网页单次、三检索器与 Top-k 对照 | `app.py` | 现有实现可定位；本轮未完成所有界面的录像验收 |
| CPU 预览与双生成后端 | `rag.py`、`app.py` | 本地 Ollama 本轮实测；AUP 是独立平台路径 |
| 来源、原文、范围及资料筛选分页 | `app.py` | 来源用于人工核查，不保证每句输出都受支持 |
| 运行记录与 CSV 导出 | `rag.py`、`evaluate.py`、`app.py` | 本轮 CLI 已真实导出；网页刷新前需保存 |
| 正式实验冻结、恢复与复算 | `condition_*.py`、`retrieval_*.py`、`experiments/` | 保留失败；历史第二轮依赖本机完整归档 |

上表按主要职责合并功能，没有把同一控件拆成多个功能点。它不替代外部试用。

## 仍需完善的证据

1. 由非开发者在明确的环境和前置条件下完成安装或启动、真实问题、来源核对及导出，记录实际耗时和失败。
2. 对 AI 槽位标签完成真实人工复核，再考虑修订语义评分；现有统计一致性不等于标签正确。
3. 校准无关问题拒答与引用支持性，优先处理来源附近出现无依据解释的情况。
4. 将历史实验的固定本机路径改为显式输入配置，需要另行进行代码修改与受影响测试；本轮尚未实施。

第三方资料、权重和许可见 [ATTRIBUTION](../data/ATTRIBUTION.md)。本页不包含个人信息、课程报告、声明签名、本机路径或完整服务日志。
