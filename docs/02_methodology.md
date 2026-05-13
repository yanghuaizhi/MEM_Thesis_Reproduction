# 02: 方法学 — 11 校准方法

> 11 个校准方法的算法说明 + 论文引用 + 本项目实现位置（plan §C.3）。

## 1. 方法分类

| 类型 | 方法 | 使用 u 信号 | 算法核心 |
|------|------|------------|---------|
| 统计 | Platt, IR, HB | 否 | sklearn 经典方法 |
| 神经基线 | UMNN, NeuCalib, DESC, SBCR | 否 | 神经校准基线 |
| 论文核心 | UMC, UMC-WOR, UAMCM, UAMCM-WOR | 是 | u=σ² 不确定度调制 |

## 2. 统计方法（type=statistical, entry=train_sta）

### Platt Scaling
- **算法**: sigmoid 逻辑回归 on logits
- **超参**: 无
- **入口**: `UMC/train_sta_{ali,avazu,criteo}.py`
- **引用**: Platt (1999), *Advances in Large Margin Classifiers*

### IR (Isotonic Regression)
- **算法**: 非参数单调回归
- **超参**: 无
- **引用**: Zadrozny & Elkan (2002), KDD

### HB (Histogram Binning)
- **算法**: 分桶常数化（M=20 个桶）
- **超参**: 无（论文默认配置）
- **引用**: Zadrozny & Elkan (2001), ICML

## 3. 神经基线（type=neural_baseline, entry=train_neu, uses_u=False）

### UMNN
- **算法**: 不约束单调神经网络（积分形式）
- **超参**: lr=1e-3, epochs=20, patience=5
- **引用**: Wehenkel & Louppe (2019), NeurIPS

### NeuCalib
- **算法**: 通用神经校准基线
- **超参**: 同 UMNN

### DESC (Deep Ensemble Shape Calibration)
- **算法**: 深度集成形状校准
- **超参**: 同 UMNN

### SBCR (Self-Boost Calibration with Ranking)
- **算法**: 自提升 + ranking loss
- **超参**: 同 UMNN + scl_lam=1e-2, scl_beta=0.95

## 4. 论文核心方法（type=paper_core, entry=train_neu, uses_u=True）

### UMC (Unconstrained Monotonic Calibration with Ranking)
- **算法**: u=σ² 单调校准 + ranking loss
- **关键超参**:
  - rescaling=True
  - u_mode=pe (PackedEnsemble σ² 派生)
  - u_use_norm=True, u_clip=[-4.0, 4.0]
- **引用**: 杨怀志 & 张晨 (2026), 清华 MEM 学位论文
- **代码**: `UMC/calib/MonotonicNN.py` `class UMC`

### UMC-WOR (UMC without ranking)
- **算法**: UMC 关闭 ranking loss 的消融
- **关键超参**: rescaling=False

### UAMCM (Uncertainty-Aware Monotonic Calibration with M)
- **算法**: u-aware 3D 积分校准（p, h_ctx, u 三维）
- **关键超参**:
  - rescaling=True
  - integral_dim=3
  - u_min=-20, u_max=20
  - alpha_max=1.0, delta_scale_init=0.1
- **代码**: `UMC/calib/MonotonicNN.py` `class UAMCM`

### UAMCM-WOR (UAMCM without ranking)
- **算法**: UAMCM 关闭 ranking 消融

## 5. PackedDeepFM Backbone（plan §C.1）

所有 11 方法共用同一个 backbone：

- **M = 16**（不可改，u=σ² 依赖 ensemble）
- **batch_size_pretrain**: AliCCP/Criteo 32K, Avazu 16K
- **lr_pretrain**: 5e-4
- **dropout**: 0.1
- **init_std**: 1e-4
- **l2_reg**: 1e-5
- **embedding_dim**: 16
- **hidden_units**: [512, 256, 128, 64]
- **pretrain_seed**: 1024（固定，因 backbone checkpoint 文件名编码了 seed）

**引用**:
- DeepFM: Guo et al. (2017), IJCAI
- PackedEnsemble: Laurent et al. (2023), AAAI

## 6. u 信号定义

`u = log(σ²)` 其中 `σ² = sigma2_epistemic`（PackedDeepFM 16 个子模型预测的认识不确定度）。

- AliCCP: u 与 p 强负相关（模式 A）
- Criteo: u 局部显著（模式 B）
- Avazu: u 方向混乱（模式 C）

## 7. shuffled-u 消融定义

`u_mode=shuffled` 时：
```python
g = torch.Generator().manual_seed(calib_seed)
u_permuted = u[torch.randperm(len(u), generator=g)]
```

确保打乱后 |Pearson(u, u_permuted)| < 0.01（sanity_check 强制审计）。
