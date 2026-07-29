#!/usr/bin/env bash
set -euo pipefail

# Launch the bbox-protocol comparison browser (pre-rendered PNGs only; no BOS, no GPU).
# Port 8507 so it collides with none of run_viewer.sh (8503), run_build_viewer.sh (8504)
# or run_inspect_viewer.sh (8506).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PHANTOM_CANVAS_DATASET="${PHANTOM_CANVAS_DATASET:-/mnt/pfs/data/yuanze/phantom_koala_inspect100_v1}"
export PHANTOM_CANVAS_OUT_ROOT="${PHANTOM_CANVAS_OUT_ROOT:-_canvas}"
exec python -m streamlit run "${ROOT}/src/phantom_data/canvas_viewer.py" \
  --server.address "${PHANTOM_CANVAS_VIEWER_HOST:-0.0.0.0}" \
  --server.port "${PHANTOM_CANVAS_VIEWER_PORT:-8507}" \
  --server.fileWatcherType poll \
  --browser.gatherUsageStats false
