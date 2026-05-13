#!/usr/bin/env bash
# scripts/run_main_experiments.sh — 阶段 2-4: 11 方法 × 3 数据集 × 3 seeds = 99 runs
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [run_main] launching 99 calibration tasks ==="
python3 -m reproduction.orchestrator --stage main --resume "$@"
