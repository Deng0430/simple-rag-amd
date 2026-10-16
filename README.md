# AUP 云端开发入门助手

一个运行在 **AMD AUP Learning Cloud** 上的中文资料问答网页，适合初次使用云端开发环境的用户。它从固定指南检索步骤，为回答附上原文和来源链接；也可以直接浏览和检索指南。

## 功能

- 8 条指南和 6 道示例题，涵盖 Python、虚拟环境、GPU、网页启动、文件保存和会话管理。
- 问答、来源查看、CSV 下载，以及可选的 3/6 条检索对比。
- 没有 GPU 时可使用仅检索预览。

助手只根据仓库中的指南作答，不能查询账户或服务器状态，也不会代替用户执行操作。请以 AUP 页面显示的资源和配额为准。使用步骤和资料来源见[部署与使用说明](docs/aup-demo.md)。

## 在 AUP 运行

选择 **Development Environments → VSCode Server GPU**，将本仓库克隆到持久目录 `/home/jovyan`，进入项目文件夹并运行：

```bash
bash start_aup.sh
```

如已克隆仓库，停止旧进程后更新并启动：

```bash
cd /home/jovyan/simple-rag-amd && git pull --ff-only && bash start_aup.sh
```

也可以在 VS Code 中用 **File → Open Folder** 打开项目文件夹，再按 **Ctrl+Shift+B** 启动默认任务。服务监听 8501，请从 Code Server 的 **PORTS 面板**打开转发地址。首次模型加载会花一些时间；网页启动不负责启动或关闭 AUP 资源。

启动脚本检查 GPU 并安装项目依赖，复用平台预装的 ROCm PyTorch。首次问答时加载 `Qwen/Qwen3-4B-Instruct-2507`。不要用普通 PyPI PyTorch 替换平台提供的 ROCm 构建。

## 本地预览

仅检查页面和检索时无需 GPU 或模型：

```bash
python -m pip install "streamlit>=1.40,<2" "scikit-learn>=1.5,<2"
python -m streamlit run app.py
```

打开本地地址后选择“仅检索预览”。此模式不生成模型回答，token 字段留空。

## 评测和本地结果

运行 6 道中文示例题，或用预览模式仅检查检索：

```bash
python evaluate.py --output runs/aup_demo.csv
python evaluate.py --preview --output runs/aup_demo_preview.csv
python -m unittest discover -s tests -v
```

AUP GPU 环境使用 `.venv/bin/python`。批量推理前请停止网页进程，避免同时加载两个模型。评测 CSV 和环境快照写入 `runs/`；汇总脚本需要显式指定 CSV 文件：

```bash
python summarize_results.py runs/aup_demo.csv
```

结果、环境快照、模型权重、虚拟环境和缓存保留在本地。网页下载的 CSV 也请保存在本地目录。

检索使用字符 2—4 gram TF-IDF，低于相关性门槛的材料会被过滤，所以实际检索数量可能少于 3 或 6。字面相似度和引用不能保证答案正确；耗时不包括模型下载、加载、预热或网页渲染。

## SQuAD 检索示例

仓库附有从 SQuAD 1.1 开发集整理的小型英文示例数据，可选用作检索流程练习。重新运行时，输出保存在本地 `runs/`：

```bash
python evaluate.py --data data/squad_demo.json --top-k 3 6 --preview --output runs/squad_preview.csv
```

## 文件与许可

| 文件 | 用途 |
| --- | --- |
| `app.py`、`rag.py` | 网页、检索、回答与 CSV 导出 |
| `data/aup_faq.json` | 中文指南、来源及示例问题 |
| `data/squad_demo.json` | SQuAD 检索示例 |
| `evaluate.py`、`summarize_results.py` | 本地评测及指定 CSV 的汇总 |
| `start_aup.sh`、`.vscode/tasks.json` | AUP 环境检查与快捷启动 |
| `docs/aup-demo.md` | 部署、使用和来源说明 |
| `data/ATTRIBUTION.md` | 数据来源和许可说明 |

代码与原创指南使用 MIT。SQuAD 数据采用 CC BY-SA 4.0；外部文档、模型和商标遵循各自条款。模型权重不包含在仓库中。

## 参考

- [AUP 官方使用指南](https://amdresearch.github.io/aup-learning-cloud/user-guide/platform-basics.html)
- [AUP Code Server 指南](https://amdresearch.github.io/aup-learning-cloud/user-guide/code-server-guide.html)
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [资料署名与来源](data/ATTRIBUTION.md)
