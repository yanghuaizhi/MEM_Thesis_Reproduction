#!/usr/bin/env bash
# scripts/run_v10_ablation.sh — 阶段 6: u_mode 消融
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
    echo "=== [run_v10] launching with user-supplied --parallel ==="
    python3 -m reproduction.orchestrator --stage v10 --resume "$@"
else
    echo "=== [run_v10] launching v10 u_mode ablation (default --parallel $PARALLEL_DEFAULT) ==="
    python3 -m reproduction.orchestrator --stage v10 --resume --parallel $PARALLEL_DEFAULT "$@"
fi
