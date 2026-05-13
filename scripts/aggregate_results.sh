#!/usr/bin/env bash
# scripts/aggregate_results.sh — 阶段 7: sanity_check + tables + figures
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [aggregate] step 1/3: sanity_check (M.0 quality gate) ==="
python3 -m reproduction.analysis.sanity_check \
    --md-out results/diff_audit/sanity_check_report.md \
    --json-out results/diff_audit/sanity_check.json

echo "=== [aggregate] step 2/3: generate tables ==="
for t in table_4_1 table_4_2 table_4_3 table_4_4 tables_5_3_5_6 table_5_4_threshold; do
    python3 -m reproduction.analysis.tables.$t
done

echo "=== [aggregate] step 3/3: generate figures ==="
for f in fig_3_pcoc_u_dist fig_3_heatmap fig_4_main fig_4_2_shuffled; do
    python3 -m reproduction.analysis.figures.$f
done

echo "=== [aggregate] DONE — results in results/{tables,figures,diff_audit}/ ==="
