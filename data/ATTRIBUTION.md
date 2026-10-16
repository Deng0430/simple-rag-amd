# Data attribution

## AUP onboarding demo (default)

`aup_faq.json` contains 8 short Chinese guides and 6 example questions authored
for this course project. The guides paraphrase platform instructions and describe
this repository's own deployment experience. They are not copied official manuals
and do not represent an official AUP support service.

Each guide records its scope, source links, source type (official documentation or
project observation), and check date. Official AUP documentation supports resource
selection, port forwarding, storage paths and session controls. Python, PyTorch and
Streamlit documentation supports the corresponding library behavior. The absence
of the `python` command and the successful `virtualenv` fallback are environment-specific
observations, not claims about every AUP image.

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
