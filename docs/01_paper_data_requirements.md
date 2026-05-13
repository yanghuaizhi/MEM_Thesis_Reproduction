# 01: 论文数据需求映射（plan §A.2）

> 论文《基于误差诊断的互联网广告点击率预估校准策略决策研究》v1.13 每个数据点的来源。
> 复现项目按本表组织实验，确保覆盖所有需要验证的数值。

## 1. 论文核心论断（5 个 P + 3 个 S，plan §A.4.1）

| # | 论断 | 关键判据 |
|---|------|---------|
| P1 | 三种误差模式可区分 | AliCCP A强过预测 / Criteo B弱欠预测 / Avazu C非单调混合 |
| P2 | AliCCP UAMCM 改善 ECE + shuffled-u 显著恶化 | 改善 ≥ 5% & shuffled ≥ +30% |
| P3 | Criteo UAMCM 改善 + shuffled-u 恶化 + 压制统计方法 | 改善 ≥ 25% & shuffled ≥ +30% |
| P4 | **Avazu UAMCM shuffled-u 未显著恶化（关键反例）** | shuffled ∈ [-σ, +σ] |
| P5 | 诊断预判 = 实验验证（论证支点） | 3/3 命中 |
| S1 | Criteo 上统计方法首选 | IR/Platt/HB 进 top-3 |
| S2 | AliCCP 上 UAMCM 谨慎推荐 | seed 一致性 ≥ 1/3 |
| S3 | Avazu 上不引入 u | 同 P4 |

## 2. Ch3 诊断数据

| 论文位置 | 数据点 | 来源（10_research_archive） | 本项目对应 |
|---------|-------|--------------------------|----------|
| Ch3 全局 PCOC | AliCCP 1.483 / Criteo 0.962 / Avazu 1.062 | `ckpt/v9_error_analysis/samples/*.npz` | v9 stage → `experiments/v9_samples/` |
| Ch3 per-u-bin | 过预测桶 18/20, 0/20, 14/20 | 同上 | `diff_with_paper.compute_per_u_bin_pcoc` |
| Ch3 PCOC CV | 24.25% / 4.31% / 7.73% | 同上 | 同上 |
| Ch3 图 3-1~3-4 | PCOC-u 分布图 | samples.npz | `figures/fig_3_pcoc_u_dist` |
| Ch3 图 3-2 | AliCCP E[Y\|p,u] 10×5 热力图 | samples.npz | `figures/fig_3_heatmap` |

## 3. Ch4 主结果数据

| 论文位置 | 数据点 | 来源 | 本项目对应 |
|---------|-------|------|----------|
| 表 4-1 全部方法 ECE/AUC/LogLoss | 11 方法 × 3 数据集 mean | `ckpt/criteo/summary_all_meanstd.csv` | `tables/table_4_1` |
| ECE std | ddof=1 std | 同上（ddof=0 已换算）| `tables/table_4_1` |
| ECE 改善 % | -13.7% / -17.6% / -46.9% | mean 推算 | `tables/table_4_2` |
| 图 4-1 方法对比 | 柱状图 | summary CSV | `figures/fig_4_main` |

## 4. Ch5 消融与决策

| 论文位置 | 数据点 | 来源 | 本项目对应 |
|---------|-------|------|----------|
| 表 5-4 三重门槛 | UMC CV%, UAMCM CV%, shuffled 验证 | summary + v10_ablation2 | `tables/tables_5_3_5_6` |
| shuffled-u 恶化 | AliCCP +70.1%, Criteo +68.6%, Avazu -7.9% | `ckpt/v10_ablation2/summary/*.csv` | `diff_with_paper.check_p5_shuffled` |
| 图 4-2 u_mode 对比 | 3 种 u_mode ECE | v10_ablation2 CSV | `figures/fig_4_2_shuffled` |

## 5. 复现优先级（plan §A.5.3）

- **必须重新生成 + 独立审计**: v10 shuffled-u（核心论证支点 P5）
- **必须重新生成**: Ch4 全部 99 主结果
- **重新生成（验证）**: v9 sample-level（PCOC 计算 + per-u-bin + ANOVA）
- **保留参考**: v1.13 论文数值（不作 ground truth，仅参考对比）

## 6. 论文数据可靠性自审（plan §A.5.1）

| 来源 | 可信度 | 复现策略 |
|------|------|---------|
| v9 NPZ | 高 | 独立重生，与历史对照 |
| `summary_all_meanstd.csv` | 中 | **优先信任复现数据** |
| v10 shuffled-u CSV | 中 | **独立审计代码**（Pearson \|corr\| < 0.01 验证）|
| Ch6 S1-S3 决策映射 | 高（逻辑层）| 论断本身稳定 |
