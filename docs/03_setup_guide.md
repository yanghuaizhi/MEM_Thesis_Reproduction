# 03: 环境与配置指南

## 1. 系统要求

| 项 | 本地（macOS）| 远程（RTX 5090 容器）|
|----|-------------|------------------|
| Python | 3.10+ | 3.10+ |
| GPU | 无需 | RTX 5090 32GB |
| vCPU | 任意 | 14 |
| 内存 | 任意 | 120 GB |
| 磁盘 | ~10 GB（代码 + meta）| 80 GB shared-nvme |

## 2. 本地环境（macOS + Claude）

```bash
cd /Users/y/Research_MEM/30_reproduction
python3 -m pip install --user --break-system-packages -e .[dev]
```

或者 conda:
```bash
conda env create -f environment.yml
conda activate mem-reproduction
```

**验证**:
```bash
python3 UMC/_paths.py                     # 路径解析
python3 -m reproduction.utils.gpu         # GPU 检测（macOS 返回 cpu）
python3 -m pytest tests/                  # 单元测试
```

## 3. 远程容器（SSH + RTX 5090）

### 3.1 容器申请

- 配置: RTX 5090 单卡 32GB / 14 vCPU / 120 GB RAM
- 地域: 山东二区（2.98 元/h）
- 磁盘: shared-nvme 扩容到 80 GB

### 3.2 首次配置

```bash
ssh container
cd /root/shared-nvme/
git clone https://github.com/yanghuaizhi/MEM_Thesis_Reproduction.git 30_reproduction
cd 30_reproduction
bash scripts/setup_env.sh                 # 装依赖 + GPU 自检
```

### 3.3 数据准备

```bash
# 1. 看数据需求（下载指南）
python3 -m reproduction.data.download --all
# (按指南手动从 USTC 链接或 Kaggle CLI 下载到 data/raw/<dataset>/)

# 2. 校验 + 写 manifest
bash scripts/download_data.sh

# 3. 预处理
bash scripts/preprocess_data.sh
```

## 4. 关键环境变量

```bash
# 数据路径（不设则用 30_reproduction/data/processed）
export MEM_DATA_ROOT=/root/shared-nvme/30_reproduction/data/processed

# checkpoint 路径
export MEM_CKPT_ROOT=/root/shared-nvme/30_reproduction/experiments

# torch-uncertainty 源码路径（PackedEnsemble 依赖）
export MEM_TORCH_UNCERTAINTY_SRC=/root/shared-nvme/torch-uncertainty/src

# 状态包（远程容器）
export MEM_STATUS_PATH=/root/status.json
```

## 5. 验证清单

跑完 setup_env.sh 后：
```bash
# 项目结构
find . -maxdepth 2 -type d | sort
# 应有 24+ 子目录

# Python 包
python3 -c "from reproduction.utils import setup_seed, setup_hardware, JsonlLogger, write_status; print('OK')"

# orchestrator 干跑
python3 -m reproduction.orchestrator --stage main --dry-run | head -5

# 配置验证
python3 -m pytest tests/test_configs.py -v
```

## 6. 网络与代理

USTC 分享链接和 Kaggle 在国内可能需要代理。远程容器建议使用：
- USTC: rec.ustc.edu.cn 通常直连快
- Kaggle CLI: 配置 ~/.kaggle/kaggle.json 后 `kaggle competitions download -c ...`
- HuggingFace mirror: 如果需要某些预训练资源，用 hf-mirror.com

## 7. 故障排查

参见 [RUNBOOK.md](RUNBOOK.md) — 10 个故障应对剧本。
