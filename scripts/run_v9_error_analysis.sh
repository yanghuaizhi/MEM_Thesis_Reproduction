#!/usr/bin/env bash
# scripts/run_v9_error_analysis.sh — 阶段 5: sample-level NPZ 生成
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [run_v9] launching v9 sample-level inference ==="
python3 -m reproduction.orchestrator --stage v9 --resume "$@"
