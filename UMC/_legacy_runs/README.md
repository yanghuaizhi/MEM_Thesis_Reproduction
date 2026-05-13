# _legacy_runs/ 历史实验脚本归档

本目录归档了 v5→v10 演进过程中产生的全部实验编排脚本。**仅作历史参考，复现工作不再使用这些脚本**——新的编排由 `reproduction/orchestrator.py`（YAML 驱动）统一替代。

## 论文 v1.13 数据采纳关系（CRITICAL）

| 版本 | 论文采纳 | 必要复现？ | 历史脚本 |
|------|---------|----------|---------|
| v5_full | 否（被 v6 取代） | **跳过** | `archive/run_v5_full.py`, `archive/run_v5_avazu_repair.py` |
| v6_phase1 | 否（过渡） | **跳过** | `archive/run_v6.py`, `archive/run_stage_a.py`, `archive/analyze_v6_phase1.py` |
| **v7** | **是（主基准）** | **必须** | `run_v7.py`, `run_criteo.py` |
| **v7_supp** | **是（最终主结果）** | **必须** | `run_v7_supplement.py`, `rerun_missing.py` |
| v8_validation | **否（消融失败：DA-SCL 不收敛）** | **跳过** | `run_v8_validation.py` |
| **v9_error_analysis** | **是（Ch3 PCOC-u 图表）** | **必须** | `run_v9_error_analysis.py`, `quickstart_v9.sh` |
| **v10_ablation2** | **是（Ch5 shuffled-u 消融）** | **必须** | `run_v10_ablation2.py` |

## 复现策略

**完全不直接调用本目录的脚本**。论文需要的 4 个版本（v7+v7_supp+v9+v10）通过以下方式重新跑：

```
v7 + v7_supp → reproduction/configs/experiments/main_99.yaml
                + reproduction/orchestrator.py --stage main
v9            → reproduction/configs/experiments/v9_error_analysis.yaml
                + reproduction/orchestrator.py --stage v9
v10           → reproduction/configs/experiments/v10_ablation.yaml
                + reproduction/orchestrator.py --stage ablation
```

新编排器在以下方面更优：
- **YAML 配置驱动**：取代硬编码路径与超参字典
- **OOM-safe batch 字典**：从 run_v7.py 中的 `AVAZU_SMALL_BATCH_METHODS` 等业务逻辑迁移到 `configs/datasets/*.yaml`
- **统一 `done.flag` + `--resume`**：断点续跑
- **subprocess 调用 UMC/train_*.py**：训练入口逐字保留，不改 UMC/ 内代码

## 各脚本简介（仅备查）

| 脚本 | 行数 | 用途 | 关键参数/配置 |
|------|-----|------|--------------|
| `run_v7.py` | 506 | v7 主实验编排（14 方法 × 2 数据集 × 3 seeds，GPU 0/1 分段） | `batch_size=32K (pretrain) / 64K (calib)`, `lr_calib=1e-3`, `epochs=20`, `patience=5` |
| `run_v7_supplement.py` | 547 | v7 多 seed 补全 + 统计方法（IR/HB/Platt） | 沿用 v7 配置 |
| `run_v8_validation.py` | 209 | UAMCM_no_u_rs + DA-SCL 验证（**失败**，跳过） | 含 dascl_weight 参数 |
| `run_v9_error_analysis.py` | 459 | 为 v7 训练好的 UMC/UAMCM 生成 sample-level NPZ（y_pred_uncalib, y_true, sigma2）| 仅 inference，无新训练；保存 `samples/*.npz` |
| `run_v10_ablation2.py` | 397 | u_mode 消融：PE (原方差) / shuffled (打乱配对) / logit (logit-based 方差) × 3 数据集 × 3 seeds | `--u_mode {pe,shuffled,logit}` |
| `run_criteo.py` | 577 | Criteo 单独实验编排（OOM 防护 + 分批策略） | Criteo 特有 batch 调节 |
| `rerun_missing.py` | 198 | 容错补跑：扫描 ckpt 找未完成 task 重跑 | -- |
| `autopilot_criteo_single_gpu.sh` | shell | Criteo 单卡自动化 | -- |
| `quickstart_v9.sh` | shell | v9 快速启动 | -- |
| `archive/run_v5_full.py` | 268 | v5 初始基线（已废弃） | 6 方法 × 2 数据集 |
| `archive/run_v5_avazu_repair.py` | 240 | v5 Avazu 数据对齐 bug 修复（历史教训） | -- |
| `archive/run_v6.py` | 274 | v6 超参定型阶段 | lr_calib=1e-3 / scl_lam=1e-2 / beta=0.95 定型 |
| `archive/run_stage_a.py` | 138 | v6 stage A | -- |
| `archive/analyze_v6_phase1.py` | 415 | v6 phase1 结果分析 | -- |

## 已知坑（来自历史脚本）

复现时需要从历史脚本中提取的关键约束，**已迁移到 `reproduction/configs/`**：

1. **AVAZU_SMALL_BATCH**：Avazu 校准 batch 必须 16K（不能用默认 64K，会 OOM）
   - 源：`run_v7.py` L343 + 多处
   - 迁移到：`configs/datasets/avazu.yaml`

2. **3 seeds {1024, 2024, 3024}**
   - 源：所有 run_v*.py 中硬编码
   - 迁移到：`configs/experiments/main_99.yaml`

3. **field_index**
   - AliCCP=0, Avazu=2, Criteo=23（v6 后硬编码）
   - 迁移到：`configs/datasets/{aliccp,avazu,criteo}.yaml`

4. **Loss-best 早停**：所有 early stopping 按 LogLoss 选 epoch
   - 源：`train_neu_*.py` callbacks 配置
   - 保持不变（train_neu_*.py 内不改）

5. **ECE bins M=100**：评估必须用 M=100 桶
   - 源：`utils/metric.py` 与 run_v* 调用
   - 保持不变

## 警告

❌ **不要直接调用本目录脚本进行复现**。
❌ **不要从本目录的脚本里复制硬编码路径** `/root/shared-nvme/PAPER/`。
❌ **不要使用 v5/v6/v8 的方法变体**（已确认废弃，会污染结果）。

✅ 使用 `reproduction/orchestrator.py` + YAML 配置。
✅ 调用 `UMC/{pretrain, train_neu_*, train_sta_*}.py`（参数化版本）。
✅ 如有疑问参考 `docs/04_experiment_protocol.md`。
