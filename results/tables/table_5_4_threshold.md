# Table 5-4 (Threshold): 三重门槛通过率

三个门槛从论文目的反推:
- **T1 必要性 (Necessity)**: 诊断 Pattern A/B/C 满足 → 该数据集需要 u-calibration
- **T2 有效性 (Effectiveness)**: UAMCM ECE < UMC ECE 且统计显著 → 实际带来改善
- **T3 机制性 (Mechanism)**: shuffled-u 恶化（或 Avazu 在 ±σ 内）→ 改善来自 u

全 Pass = 该数据集 UAMCM 在原理 + 实测 + 机制 三层都成立。

注: T2 当前用 t-statistic 近似（N=3 power 弱）；待 metrics.jsonl 含 y_pred 数据后可升级为 paired bootstrap test (n=1000)。

| Dataset | T1 必要性 | T2 有效性 | T3 机制性 | Overall |
|---|---|---|---|---|
| aliccp | No data | No data | No data | 0/3 通过 — 不成立 |
| avazu | No data | No data | No data | 0/3 通过 — 不成立 |
| criteo | No data | No data | No data | 0/3 通过 — 不成立 |

_本表与原 table_5_4_metrics.md 并存（plan §A.4.2 决策依据 + 单点指标）。_
