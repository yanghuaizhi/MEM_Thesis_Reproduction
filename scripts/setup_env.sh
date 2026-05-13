#!/usr/bin/env bash
# scripts/setup_env.sh — 环境 + GPU 自检
# 远程容器开机后第一个跑的脚本。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [setup_env] project_root=$PROJECT_ROOT ==="

# 1. Python 版本
python3 --version

# 2. 安装依赖（pip install -e .）
if [ -f pyproject.toml ]; then
    pip install -e . || true
fi
pip install -q pyyaml numpy pandas scikit-learn matplotlib

# 3. GPU 自检
python3 - <<'PY'
import sys, os
sys.path.insert(0, "UMC")
sys.path.insert(0, ".")
from reproduction.utils.gpu import detect_gpu, setup_hardware
info = detect_gpu()
print("[setup_env] GPU info:", info)
setup_hardware(verbose=True)
print(f"[setup_env] torch ok: {info.get('torch_version')}")
PY

# 4. _paths 解析
python3 UMC/_paths.py

# 5. 创建运行时目录
mkdir -p experiments/runs experiments/backbones logs results/diff_audit \
         results/tables results/figures results/manifests

# 6. 状态包初始化
python3 - <<'PY'
from reproduction.utils import write_status
write_status({"phase": "setup_env_done", "ready": True})
PY

echo "=== [setup_env] DONE ==="
