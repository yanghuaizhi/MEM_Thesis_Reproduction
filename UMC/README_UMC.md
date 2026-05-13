# UMC/ 算法代码层

本目录是 30_reproduction 项目的**算法代码层**，提供 PackedDeepFM 基座模型与 11 种校准方法的完整实现。

## 来源与归属

本目录代码基于以下两个来源融合：

1. **上游 UMC（SIGIR 2025）**：[baiyimeng/UMC](https://github.com/baiyimeng/UMC)
   - 提供 DeepFM、PackedDeepFM、UMC、UMNN、NeuCalib、DESC、SBCR 等基础方法
   - 原 README 保留在 [`README.md`](README.md)（不修改）
   - License: MIT（原仓库）

2. **本人扩展（杨怀志，MEM 论文 v1.13）**：
   - **UAMCM**（Uncertainty-Aware Multi-Calibration Model）：在 UMC 单调积分框架基础上加入 u 信号（log(σ²)）作为第三维输入，实现方差条件化校准
   - **UAMCM-WOR**：UAMCM without ranking loss
   - **UMC-WOR**：UMC without ranking loss
   - **UASAC / UASAC_R**：探索性方法（论文未采纳，保留作参考）
   - 实现位置：`calib/MonotonicNN.py` 中的 `UAMCM` 类（L147-235）

## 目录结构

```
UMC/
├── README.md                 上游 baiyimeng/UMC 的原 README（保持原文）
├── README_UMC.md             本文件（说明本目录角色）
├── LICENSE                   上游 MIT
├── slide4UMC.pdf             上游 UMC 论文 slides（参考材料）
│
├── calib/                    校准方法核心算法（不改）
│   ├── MonotonicNN.py        UMC + UMNN + UAMCM + UASAC（核心）
│   ├── NeuralCalib.py        神经校准基类
│   ├── DeepEnsemShapeCalib.py  DESC
│   ├── SelfBoostCalibRank.py    SBCR
│   └── ParallelNeuralIntegral.py  单调积分实现
│
├── models/                   基础模型（不改）
│   ├── deepfm.py             DeepFM + PackedDeepFM (M=16)
│   ├── basemodel.py          训练循环 / checkpoint 抽象基类
│   ├── inputs.py             SparseFeat / DenseFeat
│   ├── callbacks.py          训练回调（EarlyStopping）
│   ├── sequence.py           序列模型（论文未用）
│   └── layers/               DNN / FM / Embedding 层
│
├── utils/                    工具（不改）
│   ├── metric.py             ECE / AUC / LogLoss / GAUC
│   ├── inputs.py             输入处理
│   └── save_samples.py       sample-level NPZ 保存（v9 依赖）
│
├── dataset/                  数据预处理参考（ipynb 保留作参考）
│   ├── aliccp_process.ipynb
│   ├── avazu_process.ipynb
│   └── download.txt          原数据下载链接
│
├── pretrain.py               PackedDeepFM 预训练入口（参数化路径）
├── train_neu_{ali,avazu,criteo}.py    神经校准训练入口（参数化路径）
├── train_sta_{ali,avazu,criteo}.py    统计校准训练入口（参数化路径）
│
└── _legacy_runs/             历史 v5-v10 编排脚本（仅参考，不再使用）
    ├── README.md             历史脚本说明
    ├── run_v7.py / run_v7_supplement.py     ← 论文 Ch4 主结果来源
    ├── run_v8_validation.py                  ← 失败的 DA-SCL 验证（跳过）
    ├── run_v9_error_analysis.py              ← 论文 Ch3 sample-level 数据
    ├── run_v10_ablation2.py                  ← 论文 Ch5 shuffled-u 消融
    ├── run_criteo.py / rerun_missing.py
    ├── autopilot_criteo_single_gpu.sh
    ├── quickstart_v9.sh
    └── archive/              v5/v6 早期探索（更早废弃）
```

## 本目录与原 baiyimeng/UMC 的差异

| 维度 | 原 baiyimeng/UMC | 本目录 |
|------|----------------|--------|
| 核心算法 (UMC/UMNN/DESC/SBCR/NeuCalib) | 原版 | **完全一致**（逐字保留） |
| UAMCM / UAMCM-WOR | 不存在 | 新增（本人原创） |
| UMC-WOR | 不存在 | 新增（消融用） |
| UASAC / UASAC_R | 不存在 | 新增（探索性，论文未采纳） |
| 训练入口（pretrain.py / train_*.py） | 硬编码 `/root/shared-nvme/PAPER/` 路径 | **参数化**：通过 env var + `reproduction/configs/paths.yaml` 读取 |
| Tier 1/2 优化挂钩 | 无 | 训练入口顶部新增 `setup_hardware()` 调用（启用 num_workers/TF32 等） |
| 实验编排（run_v7.py 等） | 在仓库顶层 | **移到 `_legacy_runs/`**，新工作走 `reproduction/orchestrator.py` |

## 使用约束（CRITICAL）

为保证复现数值与论文 v1.13 出自同一代码路径，**严格遵守以下边界**：

| 文件 | 修改限制 |
|------|---------|
| `calib/*.py` | **完全不动**（核心算法，改了会改变数值） |
| `models/*.py` | **完全不动** |
| `utils/metric.py` | 微调允许（仅确认 ddof=1 计算，不改算法） |
| `utils/save_samples.py` | **不动**（v9 依赖） |
| `pretrain.py` | **仅允许参数化路径 + 顶部新增 setup_hardware 调用** |
| `train_neu_*.py` / `train_sta_*.py` | 同上 |
| `dataset/*.ipynb` | **保留作参考**（实际使用 `reproduction/data/preprocess/*.py` 脚本化版本） |
| `_legacy_runs/*` | **不修改、不使用**（仅作历史参考） |

如必须修改 UMC/ 内的训练逻辑，必须：
1. 先在 `tests/test_path_param.py` 跑 1 epoch 对比修改前后输出（容差 1e-6）
2. 修改提交后立即更新 `docs/07_known_issues.md` 记录原因
3. 重新跑 smoke test 验证数值漂移在可控范围

## 引用

如使用本目录代码，请同时引用上游 UMC 论文与本论文：

```bibtex
@inproceedings{UMC,
  author = {Bai, Yimeng and Zhang, Shunyu and Zhang, Yang and Liu, Hu and Bao, Wentian and Yu, Enyun and Feng, Fuli and Ou, Wenwu},
  title = {Unconstrained Monotonic Calibration of Predictions in Deep Ranking Systems},
  year = {2025},
  booktitle = {SIGIR '25},
  doi = {10.1145/3726302.3730105}
}

@mastersthesis{YangMEM2026,
  author = {Yang, Huaizhi},
  title = {基于误差诊断的互联网广告点击率预估校准策略决策研究},
  school = {清华大学经济管理学院工业工程系},
  year = {2026},
  type = {工程管理硕士学位论文},
  advisor = {张晨}
}
```
