#!/usr/bin/env bash
# scripts/run_v9_error_analysis.sh — 阶段 5: sample-level NPZ 生成
# 默认 --parallel 2: 配 hardware/rtx5090.yaml num_workers=6 → 2×6+2=14 vCPU 上限
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PARALLEL_DEFAULT=2
HAS_PARALLEL=false
for arg in "$@"; do
    case "$arg" in
        --parallel|--parallel=*) HAS_PARALLEL=true ;;
    esac
done

if $HAS_PARALLEL; then
    echo "=== [run_v9] launching with user-supplied --parallel ==="
    python3 -m reproduction.orchestrator --stage v9 --resume "$@"
else
    echo "=== [run_v9] launching v9 sample-level inference (default --parallel $PARALLEL_DEFAULT) ==="
    python3 -m reproduction.orchestrator --stage v9 --resume --parallel $PARALLEL_DEFAULT "$@"
fi
