#!/usr/bin/env bash
# scripts/preprocess_data.sh — 三数据集预处理
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

for ds in aliccp avazu criteo; do
    echo "=== [preprocess] $ds ==="
    python3 -m reproduction.data.preprocess.$ds
done
