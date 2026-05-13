# 04: 实验执行 SOP（plan §H 8 阶段）

> 完整复现执行手册。**按阶段依次执行**，每个阶段写 done.flag 后下个阶段才启动。
> 失败重跑用 `--resume`。

## 阶段 0: 环境与数据（~4-7 h）

### 0.1 容器开机 + 自检（30 min）
```bash
bash scripts/setup_env.sh
```
**输出**: CUDA 可用 / PyTorch 2.7 / 依赖装好 / status.json 就绪

### 0.2 RTX 5090 Tier 2 验证（1 h GPU）
```bash
bash scripts/smoke_test_rtx5090.sh
```
**验收**: TF32 ON/OFF 的 ECE 差异 < 1e-5 → 启用 Tier 2

### 0.3 数据下载（2-4 h IO）
```bash
bash scripts/download_data.sh
```
**输出**: `data/raw/{aliccp,avazu,criteo}/` + md5 manifest

### 0.4 数据预处理（1-2 h CPU）
```bash
bash scripts/preprocess_data.sh
```
**输出**: `data/processed/{aliccp,avazu,criteo}/data.pkl` + `feature_meta.json`

## 阶段 1: PackedDeepFM Backbone（~28 h GPU）

```bash
bash scripts/run_pretrain.sh
```

9 个 backbone (3 datasets × 3 seeds)，11 个校准方法共用。

| 数据集 | 每 seed GPU 时长（@1.5x）|
|--------|--------------------|
| AliCCP × 3 | 12 h |
| Avazu × 3 | 6 h |
| Criteo × 3 | 10 h |

**验证**: `find experiments/runs/pretrain -name done.flag | wc -l` == 9

## 阶段 2-4: 校准实验（~30 h GPU）

```bash
bash scripts/run_main_experiments.sh
```

99 任务 (11 methods × 3 datasets × 3 seeds)：
- 阶段 2 统计基线 (Platt/IR/HB)：< 1 h
- 阶段 3 神经基线 (UMNN/NeuCalib/DESC/SBCR)：13 h
- 阶段 4 UMC/UAMCM 主实验：17 h

**核心论断依赖此阶段**。完成后立即跑：
```bash
python3 -m reproduction.analysis.diff_with_paper --layer method
```
检查 P2/P3/P4 三态结果（plan §M.3）。

## 阶段 5: v9 Sample-Level（~2 h GPU）

```bash
bash scripts/run_v9_error_analysis.sh
```

对每个 (dataset, seed=1024) 跑 UAMCM inference 生成 NPZ。**直接对应 Ch3**。

完成后跑:
```bash
python3 -m reproduction.analysis.diff_with_paper --layer diagnosis
```
检查 P1 三态结果（plan §M.2）。

## 阶段 6: v10 u_mode 消融（~4 h GPU）

```bash
bash scripts/run_v10_ablation.sh
```

27 任务 (uamcm × 3 u_modes × 3 datasets × 3 seeds)。**P5 论证支点依赖此阶段**。

完成后跑:
```bash
python3 -m reproduction.analysis.diff_with_paper --layer mechanism
```
检查 P5 三态（plan §M.4）。

## 阶段 7: 聚合产物（<30 min CPU）

```bash
bash scripts/aggregate_results.sh
```

依次执行:
1. `sanity_check`（M.0 内部质量门）
2. 5 个 table 脚本 → `results/tables/*.{md,csv}`
3. 4 个 figure 脚本 → `results/figures/*.{pdf,png}`

## 阶段 8: 差异审计（<10 min CPU）

```bash
bash scripts/generate_paper_artifacts.sh
```

输出:
- `results/diff_audit/L1_diagnosis_verification.md` (P1)
- `results/diff_audit/L2_method_verification.md` (P2/P3/P4)
- `results/diff_audit/L3_mechanism_verification.md` (P5)
- `results/diff_audit/L4_decision_verification.md` (S1/S2/S3)
- `results/diff_audit/diff_with_v1_13.md` (综合)

## 完成验收

跑完所有阶段后:
```bash
# 1. done.flag 计数（应有 135 = 9 + 99 + 27 个）
find experiments/runs -name done.flag | wc -l

# 2. sanity_check 必须通过
python3 -m reproduction.analysis.sanity_check --strict || echo "FAIL"

# 3. 差异审计已生成
ls results/diff_audit/

# 4. git push
git add results/ docs/05_results_summary.md docs/06_paper_diff_audit.md
git commit -m "complete reproduction run X"
git push
```

## 估时合计

| 阶段 | 时长 |
|------|------|
| 0 环境+数据 | 4-7 h |
| 1 backbone | 28 h |
| 2-4 校准 | 30 h |
| 5 v9 | 2 h |
| 6 v10 | 4 h |
| 7-8 聚合+审计 | 0.5 h |
| 失败重跑 buffer | 13 h |
| **总计** | **~83 h** (预算 114 h，余 27%) |
