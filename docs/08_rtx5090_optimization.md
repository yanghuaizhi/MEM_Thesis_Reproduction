# 08: RTX 5090 硬件优化（Tier 1 / 2 / 3）

> 来源: plan §D.1-D.2 + `reproduction/utils/gpu.py` `setup_hardware` 接口。

## 1. 性能假设（plan §D.1，修正后）

| 配置 | 相对 RTX 4090 加速 | 信心 |
|------|------------------|------|
| 基础（无优化） | 1.3x | 高 |
| + Tier 1 优化 | 1.35-1.4x | 高 |
| + TF32 显式 | **1.5-1.7x (基准)** | 中（需 smoke test）|
| + torch.compile | 1.6-1.9x (乐观) | 中（Blackwell 兼容性 verify）|

**预算重算（1.5x 基准）**: 实验 65 h + 缓冲 18 h = **83 h**，预算 114 h，余 27%。

## 2. Tier 1: 完全安全（必做）

DataLoader + eval batch 优化，不影响数值。

`reproduction/configs/hardware/rtx5090.yaml`:
```yaml
dataloader:
  num_workers: 12             # 14 vCPU 留 2
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

eval:
  batch_size_multiplier: 4    # 推理 batch = batch_size_calib * 4
```

orchestrator 自动从配置注入到 `_runner.py` → `setup_hardware()` → UMC 训练。

## 3. Tier 2: 近无关（需 smoke test）

TF32 精度切换，理论上不改 ECE，但需验证。

```yaml
precision:
  allow_tf32_matmul: true
  allow_tf32_cudnn: true
  matmul_precision: high

compile:
  enabled: false              # 默认关，验证后开
```

**验收**: `tests/test_tf32_drift.py` 在 AliCCP × seed=1024 × UMC × 2 epoch 下，
TF32 ON/OFF 的 |ΔECE| < 1e-5。

```bash
MEM_SMOKE_TEST_TF32=1 python3 -m pytest tests/test_tf32_drift.py -v
```

## 4. Tier 3: 红线（禁止动）

`reproduction/utils/gpu.py::setup_hardware` 强制 enforce，cfg 不能覆盖:

| 项 | 强制值 | 理由 |
|----|-------|------|
| cudnn.benchmark | False | 引入不确定性（plan §B 第 8 条）|
| cudnn.deterministic | True | 保证可复现 |
| 混合精度 (BF16/FP16) | False | 论文是 FP32 |
| train batch_size | 来自 datasets/*.yaml | 改了数值漂移 |
| lr / epochs / seeds | 配置 | 同上 |

## 5. GPU 监控

```bash
# 实时 nvidia-smi
ssh container 'nvidia-smi'

# 5min 周期状态包（远程容器后台跑）
bash scripts/health_check.sh &

# 本地查
ssh container 'cat /root/status.json' | jq '.gpu, .budget'
```

## 6. 预算管理

| 项 | 值 |
|----|-----|
| 单价 | 2.98 元/h |
| 总预算 | 340 元 |
| 总机时 | 114 h |
| 实验估时 | 65 h |
| 调试/失败 buffer | 18 h |
| 数据下载/预处理 | 5 h |
| 余量 | 27% |

超支预案: 阶段 1 实测后评估，若 >35 h 则降级为 1 seed 主实验（保留 v9/v10
完整 3 seeds，因 P5 论证支点强依赖多 seed CV）。

## 7. 实测加速记录

| 阶段 | 估时（h）| 实际（h）| 加速比 |
|------|---------|---------|-------|
| Backbone | 28 | — | — |
| 神经基线 | 13 | — | — |
| UMC/UAMCM | 17 | — | — |
| v9 + v10 | 6 | — | — |

跑完后填入，作为下一次复现的基准。
