# 检索实验 v2：新增 holdout（12 题）

本集为 **AI 辅助、依官方语料核对的暂定 gold，未经独立人工验证**。它用于课程 RAG 的小样本测试，不是人工专家标注的通用基准。

## 文件与使用边界

- `holdout_questions.json`：12 个新问题，保留现有 `metadata` + `questions` 结构；问题记录仅含 ID、场景 ID、类别、问题和答案类型。
- `holdout_gold.json`：采用现有评分器要求的 `metadata` + `items` 顶层结构，包含相同问题及评分专用的 `reference_answers`、`required_slots`、`allowed_extras`、`critical_errors`、`relevant_evidence`。与现有评分数据字段结构一致。
- gold 只能用于评分。检索、上下文组包、模型生成均只接收问题文本和允许的普通问题元数据，不得接收参考答案、槽位或证据标签。

## 来源与选择过程

冻结语料：`runs/condition_eval/2026-10-27_test_v1/corpus.json`。其实际 SHA-256 为：

`14b90cbbafc7940105331a8a92e276edf23bc77bad7796c9227a293e837a11b7`

选题依据是来源可回答性、操作场景差异和文档覆盖。未运行检索或生成模型，未读取候选排序或任何正式预测与结果；不依据是否命中、组间差异或预期提升选题。

已经逐项避开旧正式集 20 个场景以及开发集 3 个场景，并额外避开旧说明提到的 6 个快捷问题场景（Python 命令缺失、ensurepip、网页代理、会话持久保存、AMD cuda 接口、CPU/GPU 资源选择）。同一文档可以再次使用，但操作意图和评分约束必须不同。

暴露披露：读取旧开发集以排除场景时，因其问题与参考字段同存一个 JSON，参考字段也意外显示。没有读取旧正式 gold.json、预测或结果。因此不能宣称作者从未接触任何旧参考字段；本集每项新答案均重新由冻结官方原文支持。

## 覆盖

四类各 3 题，共 12 个新场景、12 个关键约束槽位，覆盖 9 份官方文档。类别仅用于报告，不代表难度或主题完全互斥。

| ID | 类别 | 新场景 |
| --- | --- | --- |
| rv2-01 | 路径与配置 | vscode-project-task-location |
| rv2-02 | 路径与配置 | git-existing-destination-directory |
| rv2-03 | 路径与配置 | hf-local-download-metadata-cleanup |
| rv2-04 | Python与pip | python-interpreter-environment-detection |
| rv2-05 | Python与pip | pip-global-package-version-limits |
| rv2-06 | Python与pip | pip-package-specific-stable-release-policy |
| rv2-07 | Git与HF选项 | git-clone-release-tag-checkout |
| rv2-08 | Git与HF选项 | git-pull-fast-forward-only-update |
| rv2-09 | Git与HF选项 | hf-snapshot-json-file-selection |
| rv2-10 | 云端服务与运行 | aup-remote-server-explicit-stop |
| rv2-11 | 云端服务与运行 | streamlit-origin-allowlist-activation |
| rv2-12 | 云端服务与运行 | hf-file-transfer-timeout-failure |

官方文档：aup-platform、git-clone、git-pull、hf-download、hf-env、pip-guide、python-venv、streamlit-config、vscode-tasks。

## 标注与核验

- `short_answer` 只填写每题指定的单一短目标；其余条件回答放入 `explanation`。所有实质性问题都在字面分隔标记“简短答案”之前，检索剥离格式后缀时不会丢失条件信息。
- 每题恰有一个 required slot。它是题目未直接给出的必要使用前提、适用版本、操作约束或行为说明，不把原题已给定的目录、包名等背景另行计分。
- `slot_type` 明示槽位性质；`behavior_explanation` 不能被报告成纯执行前提。本集不把每个槽位都称为前置条件。
- 参考短答案集中于命令、参数、路径、表达式或名称。别名是有限示例，不能用简单关键词命中替代语义核验。完整正确命令也应作为等价答案接受。
- 逐字引文、passage_id、来源 URL 和标题均从冻结语料核对。所有引用是对应正文的连续子串，所有槽位的 evidence_ids 均在本题 relevant_evidence 内。
- 已检查 12 个唯一问题 ID、12 个唯一场景、四类各 3 题、每题一个槽位、与旧 23 个场景 ID 无重叠及 9 份官方文档覆盖。另由作者逐项比较场景含义，避免仅更名的旧题。
- 生成前独立代理审计指出 gold 顶层键和答案格式后缀需对齐现有评分器；已将 `questions` 修为 `items`，并将条件说明明确分配给 `explanation`。场景、参考答案、槽位与证据均未改变。修订后 `condition_experiment.validate_gold` 校验通过；此时尚未读取检索排名或生成输出。
- 生成前创建并校验本版本。取得任意实验生成输出后，不得按输出修改或替换问题、答案、槽位。若发现真实标注错误，应另建显式修订版本，记录理由并对全部比较方法统一重评。
- 未执行题目中的命令，未下载示例仓库或安装包；example.com 地址只是题面占位示例。

## 生成前审计修订后的校验和

- `holdout_questions.json`：`3121c2780991a49c8e003b29b5898dfc87c1fd93afac00ca83956018d2f351d3`
- `holdout_gold.json`：`0dc672484408258c2123c020d21a2b51fa51b9991e4bada984827806e0c86cfa`

这些校验和用于核对后续实验使用同一份问题与 gold；本文件不包含模型结果。


## 冻结前独立核查

第二个代理已核验12条问题/gold结构、逐字引文、来源标题与槽位ID。`rv2-08` 的 git pull --ff-only 分叉场景与用户早期部署经历相似，因此“新题”指未进入此前20题正式集和3题开发评测，不代表所有参与者从未见过类似场景。本轮选题没有读取检索排名或新模型输出，没有因此替换该题。
