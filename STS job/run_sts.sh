#!/bin/bash
# run_sts.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

source .venv/bin/activate
echo "Starting Similarity Search Job..."
python3 sts_job_fin.py "$@"