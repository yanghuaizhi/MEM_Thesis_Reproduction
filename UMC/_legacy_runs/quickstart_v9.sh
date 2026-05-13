#!/bin/bash
# quickstart_v9.sh — 远程服务器快速启动脚本
# 用法: bash quickstart_v9.sh [verify|smoke|p0|p1|p2|all|run]
#
# 前置条件:
# 1. 已 ssh 到远程服务器
# 2. 已 git pull 最新代码

set -e

PAPER_ROOT="/root/shared-nvme/PAPER"
UMC_DIR="$PAPER_ROOT/UMC"
OUT_DIR="$PAPER_ROOT/ckpt/v9_error_analysis"

PHASE="${1:-verify}"

echo "=========================================="
echo "V9 Error Analysis Quick Start"
echo "Phase: $PHASE"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

case "$PHASE" in
  verify)
    echo ""
    echo "Step 1: Verify backbone checksums"
    echo "----------------------------------"
    cd "$UMC_DIR"
    python run_v9_error_analysis.py --verify-backbone

    echo "Step 2: Verify GPU"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"

    echo ""
    echo "Step 3: Verify data"
    for DS in aliccp avazu criteo; do
      if [ -f "$PAPER_ROOT/dataset/$DS/data.pkl" ]; then
        SIZE=$(du -h "$PAPER_ROOT/dataset/$DS/data.pkl" | cut -f1)
        echo "  $DS: $SIZE"
      else
        echo "  $DS: NOT FOUND"
      fi
    done

    echo ""
    echo "Step 4: Dry run"
    python run_v9_error_analysis.py --phase all --dry-run
    ;;

  smoke)
    echo "Smoke test (1 epoch, verify pipeline)"
    cd "$UMC_DIR"
    python run_v9_error_analysis.py --phase p1 --dataset aliccp --method umc --seed 2024 --smoke
    ;;

  p0)
    echo "Running P0: seed 1024 verification (AliCCP only)"
    cd "$UMC_DIR"
    python run_v9_error_analysis.py --phase p0
    ;;

  p1)
    echo "Running P1: core sample-level extraction (seed 2024)"
    cd "$UMC_DIR"
    python run_v9_error_analysis.py --phase p1
    ;;

  p2)
    echo "Running P2: multi-seed supplement (seeds 1024, 3024)"
    cd "$UMC_DIR"
    python run_v9_error_analysis.py --phase p2
    ;;

  all)
    echo "Running ALL phases sequentially"
    cd "$UMC_DIR"
    python run_v9_error_analysis.py --verify-backbone
    echo ""
    python run_v9_error_analysis.py --phase all
    ;;

  run)
    # 后台运行 P1 (核心实验), 防止前端断开影响
    # 用法: bash quickstart_v9.sh run
    echo "Starting P1 in background (nohup)..."
    echo "Log: $OUT_DIR/v9_master.log"
    echo "Monitor: tail -f $OUT_DIR/v9_master.log"
    echo ""

    cd "$UMC_DIR"

    # 先验证环境
    python run_v9_error_analysis.py --verify-backbone

    # 后台启动 P1 (Avazu -> Criteo -> AliCCP)
    nohup bash -c '
      cd '"$UMC_DIR"'
      echo "===== P1 START: $(date) ====="
      python run_v9_error_analysis.py --phase p1
      echo ""
      echo "===== ALL DONE: $(date) ====="
    ' > "$OUT_DIR/v9_master.log" 2>&1 &

    PID=$!
    echo "Background PID: $PID"
    echo "$PID" > "$OUT_DIR/v9_master.pid"
    echo ""
    echo "Commands:"
    echo "  tail -f $OUT_DIR/v9_master.log    # 实时查看进度"
    echo "  kill $PID                          # 终止实验"
    echo ""
    echo "P1: 6 experiments, ~5h total"
    echo "  Avazu (2 runs, ~80min) -> Criteo (2 runs, ~50min) -> AliCCP (2 runs, ~160min)"
    echo "  Avazu 完成后即可开始分析 Simpson's Paradox"
    echo "  完成后日志末尾显示 ALL DONE"
    ;;

  *)
    echo "Usage: $0 [verify|smoke|p0|p1|p2|all|run]"
    echo ""
    echo "  verify  — 验证环境 (backbone, GPU, data)"
    echo "  smoke   — 冒烟测试 (1 epoch)"
    echo "  p0      — seed 1024 验证 (~2h)"
    echo "  p1      — 核心 sample-level 提取 (~6h)"
    echo "  p2      — 多 seed 补全 (~9h)"
    echo "  all     — 全部顺序执行"
    echo "  run     — 后台运行 P0+P1 (nohup, 防断连)"
    exit 1
    ;;
esac

echo ""
echo "Done. $(date '+%Y-%m-%d %H:%M:%S')"
