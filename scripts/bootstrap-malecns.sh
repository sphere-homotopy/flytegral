#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/.vendor/doomfly"
VENV="$ROOT/.venv-malecns"
UPSTREAM_URL="https://github.com/nftechie/doomfly.git"
UPSTREAM_COMMIT="71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 is required for the pinned DOOMFLY neural environment" >&2
  exit 1
fi

mkdir -p "$ROOT/.vendor"
if [ ! -d "$VENDOR/.git" ]; then
  git clone "$UPSTREAM_URL" "$VENDOR"
fi

git -C "$VENDOR" fetch origin "$UPSTREAM_COMMIT"
git -C "$VENDOR" checkout --detach "$UPSTREAM_COMMIT"

if [ ! -d "$VENV" ]; then
  python3.11 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
  -r "$VENDOR/requirements-neural.txt" \
  -r "$VENDOR/doom/requirements.txt" \
  --build-constraint "$VENDOR/neural-build-constraints.txt"

cd "$VENDOR"
"$VENV/bin/python" - <<'PY'
from pathlib import Path
import hashlib
import json
import urllib.request

name = 'malecns_v1'
registry = json.loads(Path('doom/datasets.json').read_text())['datasets'][name]
locked = json.loads(Path(f'data-provenance/{name}/source.lock.json').read_text())
root = Path('connectome_data') / name
root.mkdir(parents=True, exist_ok=True)

for filename, url in registry['files'].items():
    target = root / filename
    if not target.exists():
        print(f'downloading {filename} ...')
        partial = target.with_suffix(target.suffix + '.download')
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    with target.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != locked[filename]['sha256']:
        raise RuntimeError(f'Source checksum mismatch: {filename}')

(root / 'source.lock.json').write_text(json.dumps(locked, indent=2) + '\n')
PY
"$VENV/bin/python" -m doom.connectome malecns_v1
"$VENV/bin/python" -m doom.prepare

echo
echo "MaleCNS graph prepared at:"
echo "  $VENDOR/outputs/doom/malecns_v1/graph.npz"
echo "Next: train the Flytegral readout with brain_runtime/train_readout.py"
