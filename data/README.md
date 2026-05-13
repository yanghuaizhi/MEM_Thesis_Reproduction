# 数据目录（不入 git，存 SSH 容器 /root/shared-nvme/data/）

本目录在 git 仓库中**不存数据**，仅记录数据来源、md5 校验值与处理流程。实际数据通过 `bash scripts/download_data.sh` 在远程容器下载。

## 数据集清单

| 数据集 | 业务场景 | 样本量（实测）| 正样本率（实测）| 特征数 | 大小（原始） | 大小（处理后） |
|--------|---------|--------|---------|--------|------------|--------------|
| AliCCP | 阿里电商 | ~17M | **3.33%** | 14（9 user + 5 item）| ~15GB | ~3GB pkl |
| Avazu  | 移动广告 | ~40M (v9 sample ~8M) | **16.31%** | 21（columns[3:] from C1）| ~5GB | ~2GB pkl |
| Criteo | 展示广告 | ~45.8M | **25.65%** | 39（13 I + 26 C）| ~11GB | ~7GB pkl |
| **合计** | -- | ~100M | -- | -- | **~31GB** | **~12GB** |

注：CTR 与样本量来自 `10_research_archive/ckpt/v9_error_analysis/samples/*.npz`
和 `10_research_archive/dataset/criteo/artifacts/data_meta.json`（实测数据，非估算）。

## 下载源（优先级排序）

### AliCCP（阿里电商点击-转化）

- **主源（USTC）**：与原 UMC 论文一致
  - URL：`https://rec.ustc.edu.cn/share/...`（详见 `reproduction/configs/datasets/aliccp.yaml`）
  - 优势：与 baiyimeng/UMC 实验一致，无字段偏差
- **备源（天池）**：https://tianchi.aliyun.com/dataset/408
- **预期 md5**：（首次下载后写入 `manifests/data_md5.txt`）

### Avazu（移动广告）

- **主源**：Kaggle Avazu CTR Prediction
  - https://www.kaggle.com/c/avazu-ctr-prediction/data
- **备源（学术镜像）**：USTC（同 AliCCP）

### Criteo（展示广告）

- **主源**：Criteo 1TB Click Logs（用前 1/N 切片，与论文一致）
  - https://ailab.criteo.com/criteo-1tb-click-logs-dataset/
- **备源**：Kaggle Criteo Display Advertising Challenge

## 目录结构（数据下载后）

```
data/
├── raw/                          # 原始数据（不入 git）
│   ├── aliccp/
│   │   ├── ali_ccp_train.txt
│   │   ├── ali_ccp_test.txt
│   │   └── ...
│   ├── avazu/
│   │   └── train.csv
│   └── criteo/
│       └── train.txt
│
├── processed/                    # 预处理后（不入 git）
│   ├── aliccp/
│   │   ├── train.parquet         # 训练集
│   │   ├── val.parquet           # 验证集（校准训练用）
│   │   ├── test.parquet          # 测试集（评估用）
│   │   └── feature_meta.json     # 特征 schema + field_index
│   ├── avazu/
│   └── criteo/
│
└── README.md                     # 本文件（入 git）
```

## 数据划分约定

| 数据集 | 训练集 | 验证集（校准训练）| 测试集（评估） |
|--------|--------|----------------|--------------|
| AliCCP | 70% | 15% | 15%（按论文一致）|
| Avazu  | 70% | 15% | 15% |
| Criteo | 70% | 15% | 15% |

具体划分逻辑见 `reproduction/data/preprocess/{aliccp,avazu,criteo}.py`。

## md5 校验

下载后执行：
```bash
bash scripts/download_data.sh  # 内含 md5 自动校验
```

校验结果写入 `results/manifests/data_md5.txt`，纳入 git 作为复现凭证。

## 与原 UMC 数据集差异

| 项 | 原 UMC（baiyimeng）| 本项目 | 差异说明 |
|----|------------------|-------|---------|
| AliCCP 下载源 | USTC | USTC | 一致 |
| field_index | aliccp=0 / avazu=2 / criteo=23 | 同 | 一致（v6 后硬编码） |
| 预处理 | dataset/*_process.ipynb | reproduction/data/preprocess/*.py（脚本化版本） | 逻辑一致，仅形式不同 |

如发现复现时数据 md5 与预期不符，参考 `docs/07_known_issues.md` §数据下载失败 部分。
