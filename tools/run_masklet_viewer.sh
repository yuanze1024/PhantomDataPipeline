#!/usr/bin/env bash
set -euo pipefail

# Launch the masklet viewer. Reads a segmented.jsonl manifest and renders
# frame/mask/ref triplets with bounding boxes. No BOS and no GPU required.
# Port 8512 (next after 8508 gate_viewer).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PHANTOM_MASKLET_DATASET="${PHANTOM_MASKLET_DATASET:-/mnt/pfs/data/yuanze/phantom_koala_newbox_v1}"
export PHANTOM_MASKLET_MANIFEST="${PHANTOM_MASKLET_MANIFEST:-segmented.jsonl}"
exec python -m streamlit run "${ROOT}/src/phantom_data/masklet_viewer.py" \
  --server.address "${PHANTOM_MASKLET_VIEWER_HOST:-0.0.0.0}" \
  --server.port "${PHANTOM_MASKLET_VIEWER_PORT:-8512}" \
  --server.fileWatcherType poll \
  --browser.gatherUsageStats false
