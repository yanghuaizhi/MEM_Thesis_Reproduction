# MEM_Thesis_Reproduction

> 清华 MEM 学位论文《基于误差诊断的互联网广告点击率预估校准策略决策研究》（作者：杨怀志｜导师：张晨）的完整实验复现项目。

## 项目目标

用严谨的实验流程**独立复现**论文 v1.13 的全部实验数据（11 校准方法 × 3 数据集 × 3 seeds = 99 主结果 + Ch3 PCOC-u 诊断 + Ch5 shuffled-u 消融），并以复现结果为依据更新论文 v1.14。

**复现哲学**：verify（独立验证），不是 reproduce-to-match（数值匹配）。详见 [`docs/04_experiment_protocol.md`](docs/04_experiment_protocol.md)。

## 一键复现

在配备 RTX 5090（或同等 32GB GPU）的容器上：

```bash
git clone https://github.com/yanghuaizhi/MEM_Thesis_Reproduction.git
cd MEM_Thesis_Reproduction

# 阶段 0：环境与数据（4-7h）
bash scripts/setup_env.sh
bash scripts/smoke_test_rtx5090.sh
bash scripts/download_data.sh
bash scripts/preprocess_data.sh

# 阶段 1-6：实验（~65h GPU @ RTX 5090）
bash scripts/run_pretrain.sh                # 9 backbones, ~28h
bash scripts/run_main_experiments.sh        # 99 主实验, ~31h
bash scripts/run_v9_error_analysis.sh       # sample-level NPZ, ~2h
bash scripts/run_v10_ablation.sh            # u_mode 消融, ~4h

# 阶段 7-8：聚合 + 论文图表 + 差异审计
bash scripts/aggregate_results.sh
bash scripts/generate_paper_artifacts.sh
python -m reproduction.analysis.diff_with_paper
```

## 主结果摘要（待复现填充）

> 复现完成后，本表将由 `reproduction/analysis/diff_with_paper.py` 自动填充。

| 数据集 | UMC ECE | UAMCM ECE | ECE 降低 | Seed 一致性 | 误差模式 |
|--------|--------|-----------|---------|-----------|---------|
| AliCCP | _ | _ | _ | _ | A 强过预测 |
| Avazu  | _ | _ | _ | _ | C 非单调混合 |
| Criteo | _ | _ | _ | _ | B 弱欠预测 |

论文 v1.13 当前数值见 [`docs/01_paper_data_requirements.md`](docs/01_paper_data_requirements.md)，作为对照参考。

## 与论文 v1.13 章节 mapping

| 论文章节 | 复现产物 |
|---------|---------|
| Ch3 表 3-7 + 图 3-1~3-4 | `results/tables/table_3_7.md` + `results/figures/fig_3_*.pdf` |
| Ch4 表 4-1 主结果 | `results/tables/table_4_1_main_results.md` |
| Ch4 表 4-2 ECE 降低 | `results/tables/table_4_2_ece_drop.md` |
| Ch5 表 5-3~5-6 决策框架 | `results/tables/tables_5_3_5_6_decision.md` |
| 摘要 / Ch6 §6.1~6.3 论断 | `results/diff_audit/diff_with_v1_13.md`（5 个 P 论断 + S1-S3 评估）|

## 项目结构

```
30_reproduction/
├── UMC/                 算法代码层（基于 baiyimeng/UMC，含 UAMCM/UASAC 扩展）
├── reproduction/        复现工作层（编排 / 配置 / 数据 / 分析 / 工具）
├── scripts/             Shell 入口
├── tests/               单元 / smoke 测试
├── docs/                完整文档
├── results/             复现产物（git 入库）
├── data/                数据（gitignored，存 /root/shared-nvme）
└── experiments/         运行时产物（gitignored）
```

## 文档导航

- [`docs/01_paper_data_requirements.md`](docs/01_paper_data_requirements.md) — 论文 v1.13 数据需求清单
- [`docs/02_methodology.md`](docs/02_methodology.md) — 11 方法算法说明 + 引用
- [`docs/03_setup_guide.md`](docs/03_setup_guide.md) — 环境搭建 + 数据下载
- [`docs/04_experiment_protocol.md`](docs/04_experiment_protocol.md) — 实验 SOP + 复现哲学
- [`docs/05_results_summary.md`](docs/05_results_summary.md) — 复现结果总览
- [`docs/06_paper_diff_audit.md`](docs/06_paper_diff_audit.md) — 与 v1.13 差异审计
- [`docs/07_known_issues.md`](docs/07_known_issues.md) — 避坑清单 10 条
- [`docs/08_rtx5090_optimization.md`](docs/08_rtx5090_optimization.md) — RTX 5090 三层优化
- [`docs/09_local_ssh_workflow.md`](docs/09_local_ssh_workflow.md) — 本地-SSH 协作模式
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — 故障处置剧本（10 个场景）

## 引用

本项目基于：

- 论文：杨怀志, 张晨. (2026). 基于误差诊断的互联网广告点击率预估校准策略决策研究. 清华大学经济管理学院工业工程系硕士学位论文.
- 上游 UMC: Bai, Y. et al. (2025). *Unconstrained Monotonic Calibration of Predictions in Deep Ranking Systems*. SIGIR 2025. https://github.com/baiyimeng/UMC

## License

MIT License — 详见 [LICENSE](LICENSE)

## 状态

**当前阶段**：项目骨架搭建中。实验尚未执行。

最后更新：2026-05-13
