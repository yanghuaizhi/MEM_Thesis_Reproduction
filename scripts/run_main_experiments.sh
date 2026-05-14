#!/usr/bin/env bash
# scripts/run_main_experiments.sh — 阶段 2-4: 11 方法 × 3 数据集 × 3 seeds = 99 runs
# 默认 --parallel 2: 配 hardware/rtx5090.yaml num_workers=6 → 2×6+2=14 vCPU 上限
# 覆盖示例: bash scripts/run_main_experiments.sh --parallel 1
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
    echo "=== [run_main] launching with user-supplied --parallel ==="
    python3 -m reproduction.orchestrator --stage main --resume "$@"
else
    echo "=== [run_main] launching 99 calibration tasks (default --parallel $PARALLEL_DEFAULT) ==="
    python3 -m reproduction.orchestrator --stage main --resume --parallel $PARALLEL_DEFAULT "$@"
fi
