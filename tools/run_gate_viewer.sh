#!/usr/bin/env bash
set -euo pipefail

# Launch the keep/drop browser. Reads gate_report.json + the stored frames; draws boxes
# itself, so no BOS and no GPU.
# Port 8508: 8503 (viewer), 8504 (build_viewer), 8506 (inspect), 8507 (canvas) are taken.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PHANTOM_GATE_DATASET="${PHANTOM_GATE_DATASET:-/mnt/pfs/data/yuanze/phantom_koala_inspect100_v1}"
export PHANTOM_GATE_OUT_ROOT="${PHANTOM_GATE_OUT_ROOT:-_redetect100}"
exec python -m streamlit run "${ROOT}/src/phantom_data/gate_viewer.py" \
  --server.address "${PHANTOM_GATE_VIEWER_HOST:-0.0.0.0}" \
  --server.port "${PHANTOM_GATE_VIEWER_PORT:-8508}" \
  --server.fileWatcherType poll \
  --browser.gatherUsageStats false
