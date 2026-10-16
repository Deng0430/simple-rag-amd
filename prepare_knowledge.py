"""Download only the public pages in the manifest and index their main content."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from bs4 import BeautifulSoup

from knowledge import CORPUS, MANIFEST, chunk_section


def extract_sections(html):
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article.bd-article, article, main, #main, div.body[role='main']")
    if article is None:
        raise ValueError("未找到文档正文，未将导航或整页 HTML 当作资料。")
    for node in article.select("script,style,nav,aside,footer,button,.headerlink,.toc,.on-this-page"):
        node.decompose()
    title, anchor, blocks = "正文", "", []
    sections = []
    # A pre block is one unit; its nested code must not be repeated as prose.
    for element in article.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li", "tr", "dt", "dd"]):
        if element.find_parent(["pre", "li", "tr", "dd"]):
            continue
        if re.match(r"h[1-4]", element.name):
            if blocks:
                sections.append((title, anchor, "\n\n".join(blocks)))
            title, anchor, blocks = element.get_text(" ", strip=True), element.get("id", ""), []
            if not anchor and element.parent.name == "section":
                anchor = element.parent.get("id", "")
        else:
            # Syntax highlighting wraps individual tokens in spans. Inserting a
            # newline between them would turn one shell command into many lines.
            text = element.get_text().strip("\n") if element.name == "pre" else element.get_text(" ", strip=True)
            if text:
                blocks.append("```\n" + text + "\n```" if element.name == "pre" else text)
    if blocks:
        sections.append((title, anchor, "\n\n".join(blocks)))
    if sum(len(s[2]) for s in sections) < 150:
        raise ValueError("正文过短或页面需要客户端渲染，未导入空页面。")
    return sections


def fetch_source(source, refresh=False):
    folder = CORPUS.parent / "pages"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (source["id"] + ".html")
    if refresh or not target.exists():
        request = urllib.request.Request(source["url"], headers={"User-Agent": "AUP-course-RAG-demo/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(8_000_001)
            if len(body) > 8_000_000:
                raise ValueError("页面超过导入大小限制。")
            html = body.decode("utf-8")
        sections = extract_sections(html)
        target.write_text(html, encoding="utf-8", newline="\n")
    else:
        html = target.read_text(encoding="utf-8")
        sections = extract_sections(html)
    checked = datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).date().isoformat()
    passages = []
    for section_number, (title, anchor, text) in enumerate(sections):
        for number, chunk in enumerate(chunk_section(text)):
            digest = hashlib.sha256(f"{source['id']}:{section_number}:{number}".encode()).hexdigest()[:12]
            passages.append(dict(
                id="doc-" + digest, document_id=source["id"],
                title=source["title"] + " / " + title,
                text=chunk, keywords=title, category=source["category"],
                scope="官方文档原文片段；通用示例应结合实际版本，本项目部署方法以项目说明为准。",
                sources=[dict(title=source["title"] + " / " + title,
                              url=source["url"] + ("#" + anchor if anchor else ""),
                              kind="官方文档原文", checked_on=checked)],
            ))
    return dict(**source, retrieved_on=checked, sha256=hashlib.sha256(html.encode()).hexdigest(),
                section_count=len(sections), chunk_count=len(passages)), passages


def prepare(refresh=False):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    documents, passages, failures = [], [], []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [(s, pool.submit(fetch_source, s, refresh)) for s in manifest["sources"]]
        for source, future in futures:
            try:
                document, chunks = future.result()
                documents.append(document)
                passages.extend(chunks)
                print(f"OK {source['id']}: {len(chunks)} chunks", flush=True)
            except Exception as exc:
                failures.append(dict(id=source["id"], url=source["url"], error=str(exc)))
                print(f"FAILED {source['id']}: {exc}", flush=True)
    if not passages:
        raise RuntimeError("没有成功导入任何官方文档；原有缓存未覆盖。")
    successful = {d["id"] for d in documents}
    complete_aup = all(s["id"] in successful for s in manifest["sources"] if s.get("required"))
    result = dict(id=manifest["id"], built_at=datetime.now(timezone.utc).isoformat(),
                  complete_aup=complete_aup, documents=documents, passages=passages, failures=failures)
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = CORPUS.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(CORPUS)
    print(f"Imported {len(documents)} pages, {len(passages)} chunks; complete AUP user guide: {complete_aup}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    prepare(args.refresh)
