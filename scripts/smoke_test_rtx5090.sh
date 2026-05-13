#!/usr/bin/env bash
# scripts/smoke_test_rtx5090.sh — Tier 2 优化数值差异验证（plan §D.2 验收）
# 在 AliCCP × seed=1024 × UMC × 2 epoch 下，TF32 ON/OFF 的 |ΔECE| < 1e-5
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [smoke_test] TF32 drift check ==="

# 这是 placeholder — 实际执行依赖 tests/test_tf32_drift.py
if [ -f tests/test_tf32_drift.py ]; then
    python3 -m pytest tests/test_tf32_drift.py -v
else
    echo "[smoke_test] tests/test_tf32_drift.py not yet implemented"
    echo "[smoke_test] manual procedure (after data + tests ready):"
    echo "  1. python3 -m reproduction.orchestrator --stage main --dataset aliccp --method umc --seed 1024 --max-runs 1"
    echo "     (with hardware/rtx5090.yaml Tier 2 OFF)"
    echo "  2. record ECE_OFF"
    echo "  3. set allow_tf32_matmul=true, allow_tf32_cudnn=true, re-run"
    echo "  4. record ECE_ON"
    echo "  5. assert abs(ECE_OFF - ECE_ON) < 1e-5"
fi
