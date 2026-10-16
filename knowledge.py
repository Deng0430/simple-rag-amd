"""Official document corpus plus project-specific documentation, without duplicate FAQs."""

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).parent
MANIFEST = ROOT / "data/source_manifest.json"
CORPUS = ROOT / ".cache/knowledge/corpus.json"
PROJECT_GUIDE = ROOT / "docs/project-usage.md"
LEGACY_FAQ = ROOT / "data/archive/aup_faq_v2.json"


def chunk_section(text, limit=1000):
    """Keep paragraph/code boundaries where possible; cap pathological long blocks."""
    chunks, current, blocks = [], "", []
    for part in re.split(r"(```[\s\S]*?```)", text.strip()):
        if part.startswith("```"):
            code = part[3:-3].strip()
            # Code fragments remain individually fenced even when a long example is split.
            while len(code) > limit - 8:
                cut = code.rfind("\n", 0, limit - 8)
                if cut < (limit - 8) // 2:
                    cut = limit - 8
                blocks.append("```\n" + code[:cut].strip() + "\n```")
                code = code[cut:].strip()
            if code:
                blocks.append("```\n" + code + "\n```")
        else:
            blocks.extend(re.split(r"\n\s*\n", part))
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if current and len(current) + len(block) > limit:
            chunks.append(current)
            current = ""
        while len(block) > limit:
            cut = block.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = block.rfind(" ", 0, limit)
            if cut < limit // 2:
                cut = limit
            chunks.append(block[:cut].strip())
            block = block[cut:].strip()
        current = (current + "\n\n" + block).strip()
    if current:
        chunks.append(current)
    return chunks


def project_passages():
    passages = []
    text = PROJECT_GUIDE.read_text(encoding="utf-8")
    for part in re.split(r"(?m)^## ", text)[1:]:
        title, _, body = part.partition("\n")
        for number, chunk in enumerate(chunk_section(body, 750), 1):
            digest = hashlib.sha256((title + str(number)).encode()).hexdigest()[:10]
            passages.append(dict(
                id="project-" + digest, title="本项目 / " + title,
                text=chunk, category="本项目说明", keywords=title, document_id="project-usage",
                scope="本仓库运行方式；平台资源状态仍以账号页面为准。",
                sources=[dict(title="本项目使用说明 / " + title,
                              url="https://github.com/Deng0430/simple-rag-amd/blob/main/docs/project-usage.md",
                              kind="项目说明", checked_on="2026-10-16")],
            ))
    return passages


def signature():
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in (CORPUS, PROJECT_GUIDE, MANIFEST))


def load_knowledge():
    data = json.loads(CORPUS.read_text(encoding="utf-8")) if CORPUS.exists() else {
        "documents": [], "passages": [], "failures": [], "complete_aup": False,
    }
    # Legacy questions are convenient shortcuts, but their guide texts are NOT indexed.
    old = json.loads(LEGACY_FAQ.read_text(encoding="utf-8"))
    questions = [{"id": q["id"], "question": q["question"]} for q in old["questions"][:6]]
    passages = project_passages() + data["passages"]
    return dict(id="aup_documents_v1", title="AUP 官方与开发文档", passages=passages,
                questions=questions, updated=data.get("built_at", "2026-10-16")[:10],
                documents=data["documents"], failures=data.get("failures", []),
                complete_aup=data.get("complete_aup", False),
                prompt_profile="aup", max_new_tokens=384)
