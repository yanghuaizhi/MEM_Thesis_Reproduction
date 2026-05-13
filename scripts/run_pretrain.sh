#!/usr/bin/env bash
# scripts/run_pretrain.sh — 阶段 1: PackedDeepFM backbone (9 个)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [run_pretrain] launching 9 backbone tasks ==="
python3 -m reproduction.orchestrator --stage pretrain --resume "$@"
