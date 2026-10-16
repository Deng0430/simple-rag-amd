#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
aup_python="${AUP_PYTHON:-python3}"
venv_python=".venv/bin/python"

echo "Checking the platform's PyTorch and GPU..."
"$aup_python" - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}; ROCm: {torch.version.hip}")
if not torch.version.hip or not torch.cuda.is_available():
    raise SystemExit("Select an AUP GPU environment with preinstalled ROCm PyTorch.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
    echo "Creating a project environment that inherits ROCm PyTorch..."
    if ! "$aup_python" -m venv --system-site-packages .venv; then
        echo "Standard venv is unavailable; using the official PyPA virtualenv zipapp..."
        "$aup_python" - <<'PY'
from pathlib import Path
import urllib.request

target = Path(".cache/virtualenv.pyz")
target.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen("https://bootstrap.pypa.io/virtualenv.pyz", timeout=60) as response:
    target.write_bytes(response.read())
PY
        "$aup_python" .cache/virtualenv.pyz --system-site-packages --no-periodic-update .venv
    fi
fi

# Constrain torch to the platform build even if a future dependency requests it.
"$aup_python" - <<'PY'
from pathlib import Path
import torch

target = Path(".cache/aup-constraints.txt")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(f"torch=={torch.__version__}\n", encoding="utf-8")
PY

echo "Installing the web app dependencies..."
"$venv_python" -m pip install -c .cache/aup-constraints.txt -r requirements.txt

echo "Starting Streamlit on port 8501. Open it through AUP's port preview or proxy."
exec "$venv_python" -m streamlit run app.py \
    --server.address=0.0.0.0 --server.port=8501 --browser.gatherUsageStats=false
