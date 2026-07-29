#!/usr/bin/env bash
set -euo pipefail

# Launch the inspection browser (pre-rendered local PNGs only; no BOS, no credentials).
# Port 8506 by default so it collides with neither run_viewer.sh (8503) nor
# run_build_viewer.sh (8504).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PHANTOM_INSPECT_DATASET="${PHANTOM_INSPECT_DATASET:-/mnt/pfs/data/yuanze/phantom_koala_inspect_v1}"
export PHANTOM_INSPECT_OUT_ROOT="${PHANTOM_INSPECT_OUT_ROOT:-_inspect}"
exec python -m streamlit run "${ROOT}/src/phantom_data/inspect_viewer.py" \
  --server.address "${PHANTOM_INSPECT_VIEWER_HOST:-0.0.0.0}" \
  --server.port "${PHANTOM_INSPECT_VIEWER_PORT:-8506}" \
  --server.fileWatcherType poll \
  --browser.gatherUsageStats false
