# Data attribution

## AUP onboarding demo (default)

`aup_faq.json` (knowledge base `aup_faq_v2`) contains 40 short Chinese guides and
40 check questions, with six of them shown as home-page shortcuts, authored
for this course project. The guides paraphrase platform instructions and describe
this repository's own deployment experience. They are not copied official manuals
and do not represent an official AUP support service.

Each guide records its scope, source links, source type (official documentation or
project observation), and check date. Official AUP documentation supports resource
selection, port forwarding, storage paths and session controls. Python, PyTorch and
Streamlit documentation supports the corresponding library behavior. Git, pip,
VS Code and Hugging Face Hub documentation supports repository management,
dependency diagnostics, tasks and model downloads/caching. Project-specific
commands are checked against `start_aup.sh`, `.vscode/tasks.json`, `rag.py`,
`evaluate.py`, `app.py` and `.gitignore`. The absence
of the `python` command and the successful `virtualenv` fallback are environment-specific
observations, not claims about every AUP image.

The collection was expanded from eight guides on 2026-10-16. All 40 guide texts
are original summaries, not a downloaded official FAQ. Each records where the
guidance applies; library API behavior and author-proposed diagnostic steps do not
mean that those failure scenarios have all been reproduced on AUP. Historical
GPU results from `aup_faq_v1` do not validate the larger v2 collection.

The project's original guide text and example questions are provided under MIT,
like the code. Linked documentation and trademarks retain their own terms. The
source-check log and demo procedure are in [aup-demo.md](../docs/aup-demo.md).

## SQuAD baseline (archived experiment)

`squad_demo.json` contains 10 questions and 30 paragraphs sampled from the
SQuAD 1.1 development set, with seed 42. Question IDs and article titles are
preserved; the JSON structure was simplified for this course demonstration.

- Dataset: Stanford Question Answering Dataset, SQuAD 1.1.
- Authors: Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang.
- Source: https://rajpurkar.github.io/SQuAD-explorer/
- Original download: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
- License: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/
- Article paragraphs originate from Wikipedia, as described by SQuAD.

The code's MIT license does not replace the dataset license or model licenses.
Model weights are downloaded separately and are not included in this repository.
