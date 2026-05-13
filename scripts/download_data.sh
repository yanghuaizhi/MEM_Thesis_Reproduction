#!/usr/bin/env bash
# scripts/download_data.sh — 数据下载（人工指南 + md5 校验）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python3 -m reproduction.data.download --all "$@"

# 校验后写 manifest
mkdir -p results/manifests
python3 -m reproduction.data.download --verify-only \
    --write-manifest results/manifests/data_md5.txt
