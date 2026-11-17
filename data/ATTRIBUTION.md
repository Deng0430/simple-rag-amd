# Data and model attribution

## Official sources

The active knowledge base imports the main article text from the 20 public pages
listed in [source_manifest.json](source_manifest.json): the four pages of the
AUP user guide and selected Python, pip, Git, VS Code, Streamlit, PyTorch HIP and
Hugging Face Hub documentation pages. Other documentation sites are not crawled
in full.

`prepare_knowledge.py` removes navigation and scripts, preserves code blocks,
and splits articles by section and paragraph. Chunks retain their source URL,
section and retrieval date; local imports record page hashes and failures.

Downloaded HTML, extracted text and vectors are cached in `.cache/`, excluded
from Git. They are not bundled here or relicensed under the code's MIT license.
Documentation and trademarks retain their upstream terms. See the source pages
for their respective documentation licenses.

## Original runtime materials

[project-usage.md](project-usage.md) documents this repository's operating steps
and is indexed alongside the official sources. These observations are specific
to the project and do not describe every AUP image.

[example_questions.json](example_questions.json) supplies six UI shortcuts. It
contains question strings only, without reference answers or grading records.
These original project materials are provided under the repository's MIT license.

## Models

- Encoder: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), MIT.
- ONNX int8 export: [Xenova/multilingual-e5-small](https://huggingface.co/Xenova/multilingual-e5-small), MIT.
- Embedding runtime: FastEmbed / CPU ONNX Runtime, mean pooling and L2 normalization;
  inputs use `query: ` and `passage: ` prefixes.
- Cloud generator: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- Local generator and tokenizer: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct),
  served as Ollama `qwen2.5:3b` / Q4_K_M.

Models and tokenizers retain their upstream licenses. Weights and tokenizer
downloads are not distributed in this repository. The embedding downloader
records the immutable revision and hashes; the structured local runtime pins
its tokenizer revision and model digest.
