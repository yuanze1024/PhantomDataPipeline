#!/usr/bin/env bash
set -euo pipefail

# Launch the Phantom-Data browser. Reads BOS creds from $PHANTOM_BOS_AKSK
# (defaults to <repo-root>/BOS_AKSK). Point PHANTOM_DATA_DIR at the dir holding
# the two Phantom parquet files.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m streamlit run "${ROOT}/src/phantom_data/viewer.py" \
  --server.address "${PHANTOM_VIEWER_HOST:-0.0.0.0}" \
  --server.port "${PHANTOM_VIEWER_PORT:-8503}" \
  --server.fileWatcherType poll
