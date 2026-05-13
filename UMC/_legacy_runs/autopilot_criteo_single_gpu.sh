#!/usr/bin/env bash
set -euo pipefail

UMC_DIR="/root/shared-nvme/PAPER/UMC"
OUT_DIR="/root/shared-nvme/PAPER/ckpt/criteo"
BACKBONE_CKPT="/root/shared-nvme/PAPER/ckpt/data_name=criteo_model_name=packed_deepfm_batch_size=32768_dropout=0.1_init_std=0.0001_lr=0.0005_l2_reg=1e-05_seed=1024_num_estimators=16_alpha=1.0_gamma=1_.pth"
BACKBONE_LOG="/root/shared-nvme/PAPER/ckpt/data_name=criteo_model_name=packed_deepfm_batch_size=32768_dropout=0.1_init_std=0.0001_lr=0.0005_l2_reg=1e-05_seed=1024_num_estimators=16_alpha=1.0_gamma=1_.log"
PROGRESS_LOG="$OUT_DIR/autopilot_progress.log"
FULL_LOG="$OUT_DIR/autopilot_run.log"
REPORT_MD="$OUT_DIR/final_delivery_report.md"
REPORT_JSON="$OUT_DIR/final_delivery_report.json"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT_DIR"
cd "$UMC_DIR"

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "$msg" | tee -a "$PROGRESS_LOG"
}

trap 'log "AUTOPILOT FAILED at line $LINENO. Check $FULL_LOG"' ERR

log "AUTOPILOT RESUME START (single RTX 3090 / 24GB)"

if pgrep -f "python pretrain.py" >/dev/null; then
  log "Detected running pretrain.py, waiting for completion..."
  while pgrep -f "python pretrain.py" >/dev/null; do
    sleep 120
    log "pretrain.py still running..."
  done
  log "pretrain.py finished."
fi

if [ ! -f "$BACKBONE_CKPT" ]; then
  log "No Criteo backbone checkpoint found, starting pretrain..."
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python pretrain.py >>"$FULL_LOG" 2>&1
  log "pretrain.py done."
else
  log "Backbone checkpoint already exists, skip pretrain."
fi

SMOKE_CSV="$OUT_DIR/summary_criteo_gpu0_smoke.csv"
if [ -f "$SMOKE_CSV" ]; then
  log "Smoke test CSV already exists, skip."
else
  log "Running Phase 2.1 smoke test..."
  PYTHONUNBUFFERED=1 python run_criteo.py --gpu 0 --smoke --out-dir "$OUT_DIR" >>"$FULL_LOG" 2>&1
  log "Smoke test finished."
fi

log "Running Phase 2.2 full 36 experiments on single GPU (merged queue, resume-aware)..."
python - <<'PY' >>"$FULL_LOG" 2>&1
import os
import csv
import run_criteo as rc

out_dir = "/root/shared-nvme/PAPER/ckpt/criteo"
os.makedirs(out_dir, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from train_neu_criteo import trial as trial_neu_criteo
from train_sta_criteo import trial as trial_sta_criteo

trial_fns = {
    ("criteo", "neu"): trial_neu_criteo,
    ("criteo", "sta"): trial_sta_criteo,
}

queue = list(rc.GPU_QUEUES[0]) + list(rc.GPU_QUEUES[1])
summary_path = os.path.join(out_dir, "summary_criteo_gpu0.csv")

# --- Resume logic: load already-completed experiments ---
completed = set()
existing_rows = []
if os.path.exists(summary_path):
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(row["name"])
            existing_rows.append(row)
    print(f"RESUME: found {len(completed)} completed experiments in {summary_path}")
    for name in sorted(completed):
        print(f"  SKIP (already done): {name}")

# Clean up partial files from interrupted experiment
for _, exp_name in queue:
    run_name = f"criteo_{exp_name}"
    if run_name not in completed:
        for suffix in ["_ubins.csv", "_ubins_ece_best.csv"]:
            partial = os.path.join(out_dir, f"{run_name}{suffix}")
            if os.path.exists(partial):
                os.remove(partial)
                print(f"  CLEANUP partial: {partial}")
        partial_log = os.path.join(out_dir, "logs", f"{exp_name}.log")
        if os.path.exists(partial_log):
            os.remove(partial_log)
            print(f"  CLEANUP partial log: {partial_log}")
        partial_epochs = os.path.join(out_dir, "logs", f"{exp_name}_epochs.csv")
        if os.path.exists(partial_epochs):
            os.remove(partial_epochs)
            print(f"  CLEANUP partial epochs: {partial_epochs}")

summary_rows = list(existing_rows)
total = len(queue)
remaining = sum(1 for _, n in queue if f"criteo_{n}" not in completed)
print(f"RESUME: {remaining} experiments remaining out of {total} total")

for idx, (dataset, exp_name) in enumerate(queue, 1):
    run_name = f"criteo_{exp_name}"
    if run_name in completed:
        continue

    exp = rc.ALL_EXPERIMENTS[exp_name]
    trial_type = exp["trial_type"]
    if trial_type == "neu":
        config = rc.build_config_neu(exp)
    else:
        config = rc.build_config_sta(exp)
    config["uncertainty_bin_save_path"] = os.path.join(out_dir, f"{run_name}_ubins.csv")
    trial_fn = trial_fns[(dataset, trial_type)]
    row = rc.run_single_experiment(
        trial_fn=trial_fn,
        config_update=config,
        name=run_name,
        out_dir=out_dir,
        exp_idx=idx,
        total=total,
        gpu_id=0,
        dataset=dataset,
        trial_type=trial_type,
    )
    summary_rows.append(row)
    rc._append_csv(row, summary_path)

rc._write_csv(summary_rows, summary_path)

unified_path = os.path.join(out_dir, "summary_criteo_unified.csv")
with open(summary_path, newline="") as src, open(unified_path, "w", newline="") as dst:
    reader = csv.DictReader(src)
    rows = list(reader)
    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"single_gpu_full_done rows={len(summary_rows)} summary={summary_path}")
print(f"single_gpu_unified_written path={unified_path}")
PY
log "Full 36 experiments finished."

log "Running Phase 3 merge-all..."
python run_criteo.py --merge-all --out-dir "$OUT_DIR" >>"$FULL_LOG" 2>&1
log "merge-all finished."

log "Generating final delivery report..."
python - <<'PY'
import json
import re
from pathlib import Path
import pandas as pd

out_dir = Path("/root/shared-nvme/PAPER/ckpt/criteo")
backbone_log = Path("/root/shared-nvme/PAPER/ckpt/data_name=criteo_model_name=packed_deepfm_batch_size=32768_dropout=0.1_init_std=0.0001_lr=0.0005_l2_reg=1e-05_seed=1024_num_estimators=16_alpha=1.0_gamma=1_.log")
report_md = out_dir / "final_delivery_report.md"
report_json = out_dir / "final_delivery_report.json"

summary_unified = out_dir / "summary_criteo_unified.csv"
summary_meanstd = out_dir / "summary_all_meanstd.csv"
smoke_csv = out_dir / "summary_criteo_gpu0_smoke.csv"

def safe_read_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None

pretrain_auc = None
if backbone_log.exists():
    text = backbone_log.read_text(encoding="utf-8", errors="ignore")
    aucs = [float(x) for x in re.findall(r"test_auc:\s+([0-9]*\.[0-9]+)", text)]
    if aucs:
        pretrain_auc = aucs[-1]

smoke_df = safe_read_csv(smoke_csv)
unified_df = safe_read_csv(summary_unified)
meanstd_df = safe_read_csv(summary_meanstd)

failed_logs = []
for p in sorted((out_dir / "logs").glob("*.log")):
    content = p.read_text(encoding="utf-8", errors="ignore")
    if "EXPERIMENT_FAILED" in content:
        failed_logs.append(p.name)

result = {
    "pretrain_auc": pretrain_auc,
    "smoke_rows": int(len(smoke_df)) if smoke_df is not None else 0,
    "unified_rows": int(len(unified_df)) if unified_df is not None else 0,
    "meanstd_rows": int(len(meanstd_df)) if meanstd_df is not None else 0,
    "failed_experiments": failed_logs,
    "acceptance": {
        "pretrain_auc_ok": (pretrain_auc is not None and pretrain_auc >= 0.79),
        "smoke_ok": (smoke_df is not None and len(smoke_df) == 1),
        "unified_36_ok": (unified_df is not None and len(unified_df) == 36),
        "meanstd_42_ok": (meanstd_df is not None and len(meanstd_df) == 42),
        "no_experiment_failed_ok": (len(failed_logs) == 0),
    },
}
result["acceptance"]["all_pass"] = all(result["acceptance"].values())

lines = []
lines.append("# Criteo 单卡自动化交付验收报告")
lines.append("")
lines.append("## 关键结果")
lines.append(f"- pretrain test AUC: {pretrain_auc}")
lines.append(f"- smoke 行数: {result['smoke_rows']} (期望 1)")
lines.append(f"- summary_criteo_unified.csv 行数: {result['unified_rows']} (期望 36)")
lines.append(f"- summary_all_meanstd.csv 行数: {result['meanstd_rows']} (期望 42)")
lines.append(f"- EXPERIMENT_FAILED 数量: {len(failed_logs)}")
lines.append("")
lines.append("## 验收项")
for k, v in result["acceptance"].items():
    lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
lines.append("")
if failed_logs:
    lines.append("## 失败实验日志")
    for name in failed_logs:
        lines.append(f"- {name}")

report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"report_md={report_md}")
print(f"report_json={report_json}")
PY
log "Final report generated: $REPORT_MD"
log "AUTOPILOT COMPLETE"
