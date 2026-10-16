"""Fetch an immutable public ONNX snapshot with streamed, hash-checked downloads."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Lock
import urllib.request

REPOSITORY = "Xenova/multilingual-e5-small"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "onnx/model_quantized.onnx"]
_DOWNLOAD_LOCK = Lock()


def ensure_embedding_model(cache_dir):
    # Streamlit sessions can request different retrieval settings concurrently.
    with _DOWNLOAD_LOCK:
        return _ensure_embedding_model(cache_dir)


def _ensure_embedding_model(cache_dir):
    folder = Path(cache_dir) / "e5-small-onnx"
    manifest_path = folder / "snapshot.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all((folder / entry["path"]).is_file() and
               (folder / entry["path"]).stat().st_size == entry["size"] for entry in manifest["files"]):
            return folder, manifest
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPOSITORY}", timeout=30) as response:
        revision = json.load(response)["sha"]
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPOSITORY}/tree/{revision}?recursive=true", timeout=30) as response:
        tree = json.load(response)
    metadata = {item["path"]: item for item in tree if item["path"] in FILES}
    if len(metadata) != len(FILES):
        raise RuntimeError("ONNX 模型快照缺少必要文件。")
    folder.mkdir(parents=True, exist_ok=True)

    def download(name):
        entry = metadata[name]
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = entry.get("lfs", {}).get("oid")
        temporary = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(
            f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/{name}?download=true",
            headers={"User-Agent": "AUP-course-RAG-demo/1.0"},
        )
        digest, size = hashlib.sha256(), 0
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
        if size != entry["size"] or (expected_hash and digest.hexdigest() != expected_hash):
            raise RuntimeError(f"模型文件完整性检查未通过：{name}")
        temporary.replace(target)
        print(f"Embedding model: {name} ({size} bytes)", flush=True)
        return dict(path=name, size=size, sha256=digest.hexdigest())

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, FILES))
    manifest = dict(repository=REPOSITORY, revision=revision, files=files)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return folder, manifest


if __name__ == "__main__":
    model_path, info = ensure_embedding_model(Path(__file__).parent / ".cache/retrieval/models")
    print(model_path)
    print("Revision:", info["revision"])
