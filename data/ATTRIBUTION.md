# Data and model attribution

## Active corpus: official documents and project usage

The active knowledge base `aup_documents_v1` imports the main article text from
the 20 public pages listed in `source_manifest.json`: all four pages of the AUP
user guide, plus selected Python, pip, Git, VS Code, Streamlit, PyTorch HIP and
Hugging Face Hub documentation pages. It does not crawl those entire websites.

`prepare_knowledge.py` downloads only manifest pages, removes navigation and
scripts, preserves code blocks, and splits articles by section and paragraph.
Every chunk records its source URL, section and retrieval date; the local corpus
also records page hashes and import failures. “Complete AUP user guide” means
that the four required pages in this manifest were imported at that snapshot.

Third-party HTML and extracted text are cached in `.cache/knowledge/`, ignored
by Git. They are not bundled in this repository or relicensed under the code's
MIT license. Original documentation and trademarks retain their upstream terms;
see each source website for its documentation license.

`docs/project-usage.md` is original project documentation under MIT. Its six
sections cover this repository's setup, CPU preview, retrieval, timing and result
saving. Its implementation-specific observations do not describe every AUP image.
It is indexed alongside official documents to distinguish this project's setup
from generic examples.

## Archived Chinese FAQs

`archive/aup_faq_v2.json` contains the previous 40 original Chinese guides and
40 check questions. They paraphrase linked documentation and describe project
observations; they are not copied official FAQs. The original text is MIT.

These guide texts are excluded from the active corpus to avoid duplicating the
official sources. Six question strings remain as UI shortcuts, without indexing
their old answers or importing their relevance labels. Historical GPU answers for
`aup_faq_v1` do not validate the current document corpus.

## Embedding model

- Encoder: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), MIT.
- ONNX int8 export: [Xenova/multilingual-e5-small](https://huggingface.co/Xenova/multilingual-e5-small), MIT.
- Runtime: FastEmbed with CPU ONNX Runtime, mean pooling and L2 normalization.
- Inputs use the required `query: ` and `passage: ` prefixes, including for Chinese.

The downloader records the immutable snapshot revision and file hashes in the
local model manifest, verifies file sizes and available LFS SHA256 hashes, and
keeps weights and vectors under ignored `.cache/retrieval/`. They are downloaded
separately and not shipped in this Git repository. The generation model
`Qwen/Qwen3-4B-Instruct-2507` retains its own model license.

## SQuAD baseline (historical example)

`squad_demo.json` contains 10 questions and 30 paragraphs sampled from the
SQuAD 1.1 development set, with seed 42. Question IDs and article titles are
preserved; the JSON structure was simplified for this course demonstration.

- Dataset: Stanford Question Answering Dataset, SQuAD 1.1.
- Authors: Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang.
- Source: https://rajpurkar.github.io/SQuAD-explorer/
- Original download: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
- License: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/
- Article paragraphs originate from Wikipedia, as described by SQuAD.

The code's MIT license does not replace dataset, documentation or model licenses.
