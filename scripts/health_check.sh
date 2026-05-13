#!/usr/bin/env bash
# scripts/health_check.sh — 定期 nvidia-smi + 进度 + 状态包（5min 周期）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

while true; do
    python3 - <<'PY'
import json, sys, time
sys.path.insert(0, ".")
from reproduction.utils import write_status, update_phase, gpu_snapshot, disk_snapshot
from pathlib import Path

# done.flag 总数
runs = Path("experiments/runs")
n_done = 0
if runs.exists():
    n_done = sum(1 for f in runs.rglob("done.flag"))

update_phase(
    "monitoring",
    gpu=gpu_snapshot(),
    disk=disk_snapshot("/root/shared-nvme"),
    done_flags=n_done,
)
PY
    echo "[health_check] $(date +%H:%M:%S) status updated; done_flags=$(find experiments/runs -name done.flag 2>/dev/null | wc -l)"
    sleep 300
done
