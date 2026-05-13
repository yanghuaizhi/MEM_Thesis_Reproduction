#!/usr/bin/env bash
# scripts/generate_paper_artifacts.sh — 阶段 8: 四层差异审计 + 综合报告
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [paper_artifacts] running L1/L2/L3/L4 + summary ==="
python3 -m reproduction.analysis.diff_with_paper --all

echo "=== [paper_artifacts] outputs ==="
ls -la results/diff_audit/

echo ""
echo "=== Summary ==="
if [ -f results/diff_audit/diff_with_v1_13.md ]; then
    head -20 results/diff_audit/diff_with_v1_13.md
fi
