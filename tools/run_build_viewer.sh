#!/usr/bin/env bash
set -euo pipefail

# Launch the built-dataset browser (local files only; no BOS, no credentials).
# Port 8504 by default so it cannot collide with tools/run_viewer.sh on 8503.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PHANTOM_BUILD_DATASET="${PHANTOM_BUILD_DATASET:-/mnt/pfs/data/yuanze/phantom_koala_bboxref_v1}"
export PHANTOM_BUILD_INDEX="${PHANTOM_BUILD_INDEX:-phantom_pilot_v1}"
exec python -m streamlit run "${ROOT}/src/phantom_data/build_viewer.py" \
  --server.address "${PHANTOM_BUILD_VIEWER_HOST:-0.0.0.0}" \
  --server.port "${PHANTOM_BUILD_VIEWER_PORT:-8504}" \
  --server.fileWatcherType poll \
  --browser.gatherUsageStats false
