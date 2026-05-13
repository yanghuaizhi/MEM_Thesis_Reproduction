# 07: 避坑清单 + 实际踩坑增量

> 综合 plan §B 原始 10 条 + 复现过程中实际踩坑增量。

## 1. 原始避坑清单（plan §B，10 条）

| # | 坑 | 根因 | 复现时避免 |
|---|-----|------|----------|
| 1 | 多 seed 不一致（AliCCP UMC vs UAMCM 2/3）| 方差信号 u 在 Pattern A 强递减场景的有效性依赖初始化 | 强制 3 seed，报告 mean±std (ddof=1) 不掩盖 |
| 2 | Avazu 高 CV（UAMCM ECE std 4.36, CV 40%）| 数据分布跨度大 + CTR 低 + u 与误差非单调 | 分别报告每 seed；不试图改超参降 CV |
| 3 | v8 DA-SCL 失败（不收敛或恶化）| DA-SCL 权重与 u 范围不匹配 | **完全跳过 v8**，不浪费 GPU |
| 4 | Criteo seed 3024 极端值（ECE 9.23 vs seed 1024 ECE 1.11）| 极低 CTR + u 在低 CTR 缺方差 | 保留 3 seed 平均，明确标注 seed 3024 为离群 |
| 5 | Avazu OOM（UAMCM/UASAC 用 64K batch 触发）| u × M=16 ensemble × 64K 显存爆炸 | Avazu calib batch **强制 16K** |
| 6 | ddof=0 vs ddof=1 不一致 | CSV ddof=0 写入，论文报 ddof=1 | 复现脚本统一 **ddof=1** |
| 7 | Loss-best vs ECE-best 模型选择 | ECE-best 过拟合验证集 | 所有 early stopping 按 **LogLoss** 选 epoch |
| 8 | ECE bins M=100（不是常用 M=10）| 论文用 M=100 捕捉细粒度 | 评估必须 **M=100 bins** |
| 9 | field_index 数据集不一致 | aliccp=0, avazu=2, criteo=23 | 维护数据字典；运行前 sanity check AUC |
| 10 | ECE 表述 "改善 -X%" | v1.11 混合语义 | 全部用 "**降低 X%**" |

## 2. 复现项目实际踩坑（增量）

### 2.1 reproduction/utils/logging.py 与 stdlib 命名冲突

**症状**: `python3 reproduction/utils/gpu.py` 直接运行时，`import torch` 内部
`import logging` 被本地 logging.py 劫持，AttributeError。

**根因**: Python 直接运行 `foo.py` 时把 foo.py 所在目录加进 sys.path[0]。

**修复**: 重命名 `logging.py` → `jsonl_log.py`。

**预防**: `reproduction/analysis/`、`reproduction/data/` 内子模块**避免**使用
stdlib 重名（logging/json/io/time/copy/types/platform/random 等）。

### 2.2 train_neu_criteo.py L1002 的已知 bug（保留不修复）

**症状**:
- L1002 `"data_name": "aliccp"` 与文件名不符（应为 "criteo"）
- L1014 `"field_index": 0` 应为 23
- L1028 `"phase4_uncertainty_bins_aliccp.csv"` 应为 criteo

**根因**: UMC 原始代码硬编码错误（v6 阶段引入）。

**修复策略**: **保留原状不动**。orchestrator 通过 YAML 传入正确的
`data_name`/`field_index`/`uncertainty_bin_save_path` 覆盖。不动 UMC 源减少
数值漂移风险。

**verification**: `tests/test_orchestrator.py::test_config_update_critical_fields`
确保 orchestrator 输出的 config_update 字段 = YAML 配置值。

### 2.3 num_estimators 默认值与论文不符

**症状**:
- `UMC/pretrain.py` 默认 `num_estimators=8`
- `UMC/train_neu_criteo.py` 默认 `num_estimators=4`
- 论文要求 M=16

**修复**: orchestrator 强制传 `num_estimators=16`（来自 `main_99.yaml`）。

**verification**:
```bash
python3 -m pytest tests/test_configs.py::test_critical_constants -v
```

### 2.4 UMC → reproduction.utils 反向 import 路径

**问题**: plan §F.1 要求 UMC/train_*.py 在 main 开头
`from reproduction.utils.gpu import setup_hardware`，但 UMC/ 当前 sys.path
只加了 `30_reproduction/UMC/`，找不到 reproduction 包。

**修复方案（后续 task #12 完善 UMC 入口时）**: 在 UMC/_paths.py 内或
train_*.py main 开头，把 `_PROJECT_ROOT` 加进 sys.path：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reproduction.utils.gpu import setup_hardware
```

当前 `reproduction._runner` 已在 subprocess 内做了这件事
（`_setup_paths()` 把 PROJECT_ROOT 加进 sys.path）。

### 2.5 pytest mock 数据 σ=0 导致 P5 Avazu 判定 false-opposes

**症状**: mock test 中 Avazu PE 三 seed ECE 完全相同（std=0），shuffled 相同
（std=0），导致 `sigma_pct=0` 且 worsening_pct=-1.9% > 0% = not_in_sigma → opposes。

**根因**: mock 数据 unrealistic。

**修复**: tests/test_diff_with_paper.py `mock_v10_records` 给 Avazu 每个 seed
不同 ECE 值（如 [0.10, 0.13, 0.16]），模拟真实高 CV 现象。

## 3. SSH 容器运行时可能的坑

详见 [RUNBOOK.md](RUNBOOK.md) 10 个故障剧本。

## 4. 自检命令

任何怀疑配置错时优先跑:
```bash
python3 -m pytest tests/test_configs.py -v
python3 -m pytest tests/test_orchestrator.py::test_config_update_critical_fields -v
python3 -m reproduction.analysis.sanity_check
```
