#!/usr/bin/env bash
# scripts/run_v10_ablation.sh — 阶段 6: u_mode 消融
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [run_v10] launching v10 u_mode ablation ==="
python3 -m reproduction.orchestrator --stage v10 --resume "$@"
