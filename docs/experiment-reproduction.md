# 实验复核与重新运行

本仓库包含网页 demo、两轮本地小规模实验的实现，以及可公开的结果节选。
历史 AMD AUP / Qwen3 部署与本地 Ollama / Qwen2.5-3B 对照分开记录。

## 1. 在新克隆中复核已发表数字

只需 Python 标准库，不需要 Ollama、GPU、资料缓存或 `runs/`：

```bash
python experiments/verify_results.py
```

该命令从公开逐题数据复算 EM、Token-F1 和 Slot-F1 主表，并核对保存的汇总。
文件与评分口径见 [experiments/README.md](../experiments/README.md)。它复核的是已保存的回答与评分，
不会重新调用模型，也不把 AI 条件标注升级为人工确认结果。

## 2. 历史完整记录与公开节选

完整输入、下载的官方原文、源代码快照、原始响应、尝试日志、人工空白复核表和环境快照
保存在本地 `runs/`，另有本地归档；这些内容未随 GitHub 仓库发布。
两份详细实验记录中的 `analyze` 和 `postprocess.py` / `diagnose.py` 命令依赖这套完整记录，
不适用于只有公开节选的新克隆。

冻结语料为 710 段官方文档及 6 段项目说明，共 716 段；SHA-256：

`14b90cbbafc7940105331a8a92e276edf23bc77bad7796c9227a293e837a11b7`

这不是旧网页 `.cache/knowledge/corpus.json` 的哈希。正式实验额外保存章节位置信息及项目说明。
上游网页更新后，重新抓取不保证重建相同片段、引文或历史结果。

## 3. 重新生成答案的前提

先安装 `requirements.txt`，运行 `python prepare_knowledge.py` 准备资料，并启动 Ollama。
历史脚本校验固定的 `qwen2.5:3b` 模型 digest；不是任何同名模型都被接受：

`357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`

分词器由 `condition_model.py` 从固定 revision 下载，生成通过显式 ChatML 和
`/api/generate raw=true` 调用。temperature=0、seed=42、窗口 8192、输出上限 512。
这与网页的 `/api/chat` 提示入口不同。运行时固定种子仍不保证服务输出逐字相同。

第一轮脚本会依据现有 HTML 缓存重建章节位置，并核对原文切片。新抓取资料还需另行核查 gold 引文仍成立：

```bash
python condition_experiment.py prepare --split test --run-dir runs/my_condition_run
python condition_experiment.py run --run-dir runs/my_condition_run
python condition_experiment.py analyze --run-dir runs/my_condition_run
```

`analyze` 先计算词面分数；Slot-F1 需要另行按协议核查完整回答并提供 `slot_review.json`。
不能把缺少语义复核的运行当作已有三指标主表的复现。

第二轮 `retrieval_experiment.py` 是保留的历史实验入口，默认读取
`runs/condition_eval/2026-10-27_test_v1/` 下的 `corpus.json`、`questions.json` 和 `gold.json`。
**新克隆没有这个目录，不能直接运行第二轮 prepare。** 拥有历史完整归档时先恢复该目录；
仅有新抓取资料时，应在独立克隆中将第一轮 `prepare` 的输出目录设为这一依赖位置，
确认所有引文核验通过后再准备第二轮的新目录。日期形式只是脚本依赖的路径名称，
新运行应明确记录实际日期，不能将新语料或新响应冒充 2026-10-27 的历史数据。

```bash
python retrieval_experiment.py prepare --split holdout --run-dir runs/my_holdout_run
python retrieval_experiment.py run --run-dir runs/my_holdout_run
python retrieval_experiment.py analyze --run-dir runs/my_holdout_run
```

`--split regression` 使用原 20 题；`holdout` 使用新增 12 题。两套分别统计。
第二轮每个唯一请求前卸载并重载模型，以隔离本次遇到的结构化生成服务异常，首次不完整响应仅重试一次。
该入口会影响正在共享同一模型的其他本地会话，实验时应停止并发问答。耗时包含重载，不用于加速或能耗结论。

## 4. 使用边界

不要覆盖已有运行目录；题目、gold、代码、模型或资料变化后另建版本。
运行器会检查冻结哈希。历史源文件在各批 `source_snapshot/` 中，后续发布时的 Git HEAD
不能替代这些历史快照。公开的 20＋12 题均已暴露，继续研究时应将其用于开发或回归，另备验证题。

本次课程交付保留完整正负结果：第一轮未形成足够的差异输入，第二轮只在旧题上出现局部改善。
它支持对系统及失败原因的讨论，不能证明新方法具有普遍优势。
