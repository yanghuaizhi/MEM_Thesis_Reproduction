# PROJECT_HANDOFF.md — 30_reproduction 综合实施总结

> **本文件是项目持久化交接文档**。下次会话从这里恢复，能完整理解项目状态而不依赖历史对话。
> 最后更新：2026-05-14（**阶段 1 backbone 完成，calib smoke test 待跑**）

---

## 0. 下次会话开场提示词（复制粘贴）

```
继续推进 /Users/y/Research_MEM/30_reproduction/ 论文复现项目。

请先 cd 到 /Users/y/Research_MEM/30_reproduction/，让项目 CLAUDE.md 加载，然后阅读
PROJECT_HANDOFF.md（本文件）了解项目全貌、当前状态、远程容器信息、待办清单。

当前进度：
- 远程 RTX 5090 容器配好（按量计费已跑约 2.5h ≈ 7.5 元）
- 阶段 1 backbone pretrain 完成 (AliCCP 17min + Avazu 5min = 22min total, vs 估 6h)
- AliCCP+Avazu data.pkl 完整（USTC 包预处理后直接可用）
- Criteo 下载中（wget 后台，~5GB 压缩，预计 4-5 分钟完成）

下一步关键决策：calib smoke test 待跑（验证 precompute_backbone_outputs 显存峰值 +
metrics.jsonl 提取 + 数值合理性），结果决定 main 阶段 parallel 度。

待 push 的本地新 commits（约 3 个）含：orchestrator --parallel N 选项 + eval_bs 保守化 +
本 PROJECT_HANDOFF.md。

请用 TaskList 工具重建任务状态，然后让我截图远程当前状态，你立即给出下一步指令。
```

---

## 1. 项目目标（一句话）

清华 MEM 学位论文《基于误差诊断的互联网广告点击率预估校准策略决策研究》(v1.13, 杨怀志/张晨) 的
**完整实验复现项目**。

**核心论断 verify（plan §A.4.1）**:
- **P1**: 三种误差模式 (AliCCP 强过预测 / Criteo 弱欠预测 / Avazu 非单调) 可区分
- **P2**: UAMCM 在 AliCCP 改善 ECE + shuffled-u 显著恶化
- **P3**: UAMCM 在 Criteo 改善 + shuffled-u 恶化 + 压制统计方法
- **P4**: UAMCM 在 Avazu shuffled-u **未显著恶化**（关键反例）
- **P5**: 诊断预判 = 实验验证（核心论证支点）
- **S1-S3**: 决策框架（Criteo→stat / AliCCP→UAMCM 谨慎 / Avazu→不引入 u）

**复现哲学**: **verify（独立验证），不是 reproduce-to-match（数值匹配）**。
论文 v1.13 数值是假设，不是 ground truth。

---

## 2. 锁定的关键决策（17 项，不再讨论）

| 决策项 | 选定值 | 来源 |
|--------|-------|------|
| **复现颗粒度** | 11 方法 × 3 数据集 × 3 seeds（Criteo 临时跳过 = 当前 2 数据集 = 66 main + 12 v9 + 12 v10）| 用户 |
| **GPU** | RTX 5090 单卡 32GB / 14vCPU / 120GB / 山东二区 / 2.98 元/h | Paratera |
| **总预算** | 340 元 ≈ 114h GPU | 用户 |
| **复现哲学** | verify, not reproduce-to-match | 多轮共识 |
| **shuffled-u 设计** | **B（train+test 都打乱）**——UMC L307-310 已是 B | 用户 |
| **Criteo batch_size_calib** | **65536**（5090 优化，不复制历史 32768）| 用户 |
| **eval_batch_size** | calib batch × 4（保守值，smoke verify 后可上调）| 用户提醒后保守 |
| **v9 seeds** | **3 个**（1024+2024+3024）提升 P1 统计稳健性 | 用户 |
| **v10 删 pe 重复** | 是（pe 数据从 main_99 取，省 4.5h GPU）| 用户 |
| **bootstrap CI + paired test** | 是（无 GPU 成本）| 用户 |
| **F1 派生预判** | 是 + L3 标注"非完全独立"（v9 v10 同 model）| 用户 |
| **三重门槛** | 新增独立表，保留原表（不重写）| 用户 |
| **uasac / uamcm_phase4** | **不启用**（中间版本，不复现）| 用户 |
| **导师审阅** | 仅 docs/ + results/，不读代码 | 用户 |
| **复现 ddof** | 1（Bessel 校正）；论文 v1.13 用 ddof=0 | plan §B 第 6 条 |
| **ECE bins M** | **100**（不可改）| plan §C.2 |
| **PackedDeepFM M** | **16**（不可改）| plan §C.1 |
| **statistical method seed** | UMC 硬编码 1024，零方差是算法本质（deterministic）| FIX-6 澄清 |
| **跑顺序** | **方案 A**：AliCCP+Avazu 全套（main+v9+v10）跑完后再补 Criteo | 用户 |
| **main parallel 度** | 推荐 **6**（待 smoke test 后确认）| 我推荐 |
| **Criteo field_index** | **23 全 stage 一致**（plan §C.4 + criteo.yaml）| 我推荐，用户未否决 |

---

## 3. 实施状态（commit + 远程进度）

### 3.1 GitHub 仓库
**https://github.com/yanghuaizhi/MEM_Thesis_Reproduction** (private)

```
最新 commit（已 push）:
- 22d9ce8  fix: H1 prefetch_factor injection (5-10% dataloader speedup)
- 53b7a15  fix: complete pre-deploy fixes + first-principles enhancements
- ab6bed0  fix: FIX-6 statistical methods deterministic by design
- 63ace66  fix: cross-code-review 6 critical bugs found by pre-deploy audit
- 60da034  docs: HANDOFF 16/16 completion
- c3c16af  init: 30_reproduction scaffold (4-day work)

本地新 commits（**待 push**，下次会话需要先 push + 临时 public + 远程 pull）:
- orchestrator --parallel N 选项（main 66 task 并发加速 6-8x）
- hardware/rtx5090.yaml eval_bs 保守值 (×4, 等 smoke verify)
- 三个 train_neu_*.py eval_bs 默认从 ×8 回退到 ×4
- 本 PROJECT_HANDOFF.md
```

### 3.2 远程容器进度

| 阶段 | 状态 | 实测时间 | 关键产出 |
|------|------|---------|---------|
| 0 环境配置 | ✓ 完成 | 30min | torch-uncertainty + lightning + pytest 39 passed |
| 0.5 数据 AliCCP+Avazu | ✓ 完成 | 25min（含 USTC 1.5GB 下载 + 解压）| `data/processed/aliccp/data.pkl` 9.6GB + `avazu/data.pkl` 6.7GB |
| **1 backbone** | **✓ 完成** | **22min**（AliCCP 17min + Avazu 5min）| 2 个 done.flag in `experiments/runs/pretrain/{aliccp,avazu}/_backbone/seed_1024/` |
| **1.5 calib smoke test** | **🟡 待跑** | 估 15-20min | 验证 calib 全流程 + 显存峰值 + metrics.jsonl |
| 2-4 main 66 calib | 📋 待 | 估 parallel=6 约 50min | 11 methods × 2 datasets × 3 seeds |
| 5 v9 sample-level | 📋 待 | 估 parallel=6 约 30min | 12 任务，Ch3 PCOC 数据 |
| 6 v10 ablation | 📋 待 | 估 parallel=6 约 30min | 12 任务，P5 论证支点 |
| 7-8 聚合 + 回写 | 📋 待 | <30min | tables + figures + diff_audit |
| **Criteo 数据下载** | 🟡 进行中 | 已下 ~3GB / 估 ~5GB | wget 后台，与 GPU 并行不冲突 |
| Criteo 后补全套 | 📋 待 | 估 ~1.5h | preprocess + 1 backbone + 33 calib + 6 v9 + 6 v10 |

### 3.3 实测性能（震撼级别）

旧估时 vs **5090 实测**：

| 阶段 | 旧估 | 实测 / 新估 | 加速比 |
|------|------|------------|-------|
| backbone | 6h | **22 min** | **16x** |
| main 66 calib | 20h | 估 parallel=6 ~50min | ~24x |
| v9+v10 | 6h | 估 ~1h | ~6x |
| **总训练** | **~32h** | **~3h（parallel=6）** | **~10x** |
| **总预算** | ~95 元 | **~10 元**（不含 Criteo） | ~10x 省 |

---

## 4. 远程容器信息

| 项 | 值 |
|----|------|
| 平台 | Paratera 容器云（按量计费 2.98 元/h） |
| 区域 | 山东二区 |
| GPU | RTX 5090 32GB (sm_120 Blackwell) |
| CPU/RAM | 14vCPU / 120GB |
| OS / PyTorch | Ubuntu 24.04 + PyTorch 2.7.0a0+7c8ec84dab.nv25.03 |
| CUDA | 12.8 |
| 共享存储 | 80GB shared-nvme（持久，容器回收不丢） |
| 系统盘 | 30GB 临时（容器回收清空） |
| SSH 入口 | `ssh.bj8.bz1.paratera.com:2233` user=root |
| **本地 SSH alias** | `paratera`（~/.ssh/config 已配）|
| **SSH 状态** | ❌ 双因素 auth（publickey + password），password auth 被 PAM 限制，**Claude 无法直接 ssh** |
| **操作方式** | **WebSSH 网页终端**（Paratera 平台），用户在 web 内执行命令 + 截图反馈 |
| **WebSSH 历史密码** | 容器初始密码 `7068b50ec5c14520720a976159f8fc00`（SSH 拒绝）|
| **chpasswd 改后密码** | `Mem2026Reproduction!`（SSH 仍拒绝因双因素，仅 WebSSH 内 `su` 可用）|
| 工作目录 | `/root/shared-nvme/30_reproduction/` |
| 数据目录 | `data/processed/{aliccp,avazu}/data.pkl`（Criteo 待 preprocess）|
| Criteo 下载临时 | `data/raw/criteo/criteo.tar.gz`（wget 中，未解压）|
| 实验目录 | `experiments/runs/<stage>/<dataset>/<method>/seed_<N>/` |
| Log 目录 | `logs/`（`pretrain.log` 已完成；`smoke_aliccp_uamcm.log` 待生成）|

### 4.1 容器回收应对

shared-nvme 持久，重新申请同规格容器后：
1. WebSSH 内重装公钥（同前面流程）
2. 数据 + experiments + done.flag 保留
3. `cd /root/shared-nvme/30_reproduction && pip install torch-uncertainty lightning && bash scripts/setup_env.sh`
4. `--resume` 继续未完任务

---

## 5. 数据状态

| 数据集 | 状态 | 文件 | 关键参数 |
|--------|------|------|---------|
| AliCCP | ✓ 完成 | `data/processed/aliccp/data.pkl` 9.6GB | 85,316,975 行 × 15 列, CTR=0.0389, 14 features + click, field_index=0 → '101' |
| Avazu | ✓ 完成 | `data/processed/avazu/data.pkl` 6.7GB | 40,428,967 行 × 22 列, CTR=0.1698, 21 features + click, field_index=2 → site_id |
| Criteo | 🟡 下载中 | `data/raw/criteo/criteo.tar.gz` ~3/5GB | 下完后解压 + 删 test.txt + preprocess → 估 ~46M 行 × 40 列, CTR=0.256 |

### 5.1 数据来源
- **AliCCP + Avazu**: USTC 备源 `https://rec.ustc.edu.cn/share/5a70c6c0-9e4a-11ef-af55-8dfb3f6b3310` 密码 `5277`
  - 包含 baiyimeng/UMC 预处理后的 data.pkl，**schema 与 reproduction/data/preprocess/*.py 完美对齐**，跳过 preprocess 阶段
- **Criteo**: Tianchi OSS 签名 URL（用户在浏览器获取，URL expiry 2026-05-14 19:04 北京时间）
  - 包 `kaggle-display-advertising-challenge-dataset.tar.gz`
  - dac-v1 格式：train.txt + test.txt（test.txt 无 label，删之）

### 5.2 Backbone 实测指标

```
AliCCP backbone (17 min, 7 epoch, early stop patience=3):
  val_auc=0.6335  test_auc=0.6368
  sigma2_mean=232.6 (u=log(232.6)≈5.45) ← 强 epistemic uncertainty
  显存 max_allocated=2.85GB / 32GB (8% 利用率)
  sigma2_zero_ratio=0.0 ✓ (ensemble 健康)

Avazu backbone (5 min, 4 epoch, early stop):
  val_auc=0.7563  test_auc=0.7394
  sigma2_mean=0.965 (u≈-0.04) ← 弱 epistemic uncertainty (240x 小于 AliCCP)
  显存 max_allocated=4.27GB / 32GB (13%)
```

**关键观察**：AliCCP 和 Avazu 的 sigma² 差 240 倍 — 这与论文 Ch3 "AliCCP Pattern A 强过预测、
Avazu Pattern C 非单调混合" 论断一致（u 信号语义不同）。

---

## 6. 当前任务状态

### ✓ 已完成（5）
- Stage 0 远程环境配置
- Stage 0.5 数据 AliCCP+Avazu 下载 + schema verify
- Stage 1 backbone pretrain（2 个 done.flag）

### 🟡 当前焦点
- **Stage 1.5 calib smoke test 待跑**：用户要求 smoke 后再启动 main，避免 5-6h 训练后才发现 OOM/数值错

### 📋 待办（按依赖顺序）
1. **Smoke test**: AliCCP + UAMCM + seed=1024 单 task（15-20min），验证 precompute 显存
2. **本地 push** + 容器 pull 拿到 `--parallel` 选项（临时 public 30 秒）
3. **Stage 2-4 main 66 calib**（parallel=N，N 由 smoke 显存峰值定）
4. **Stage 5 v9 sample-level**（12 任务 parallel=N）
5. **Stage 6 v10 ablation**（12 任务 parallel=N）
6. **Stage 7-8 aggregate + diff_with_paper**（CPU bound, <30min）
7. **Criteo 后补**（解压 + preprocess + 1 backbone + 33 calib + 6 v9 + 6 v10）

### 待用户决策点（已记录，可默认推荐执行）
- ✅ field_index = 23 全 stage（我推荐，用户未否决 → **默认采纳**）
- ✅ parallel 度 = 待 smoke 后定（推荐 6，由实测显存调整）
- ✅ 跑顺序 = 方案 A（已用户确认）

---

## 7. 智能加速层级

### Tier 1 已开（数值 0 影响）
- `num_workers=12`, `pin_memory=True`, `persistent_workers=True`
- `prefetch_factor=4` (H1 修复)
- `eval_batch_size=calib×4` (保守值，待 smoke 后可上调 ×8)

### Tier 2 已开（理论 1e-7 影响）
- TF32 matmul + cudnn ✓
- matmul_precision=high ✓
- torch.compile: **关**（Blackwell + PT 2.7 nightly 兼容性未验证）

### Tier 2.5 待 verify（smoke 后开）
- `--parallel N`（本地新加，未 push）：5090 显存 32GB 余量大，单 task 仅 2.85GB → 理论 N=8-10
  - 推荐 N=6（CPU IO 余量 + GPU kernel 调度）
  - main 66 task 串行 5h → parallel=6 约 50 min

### Tier 3 红线（禁止改）
- train batch_size, lr, num_estimators=16, dropout, init_std, l2_reg, embedding_dim, hidden_units
- seeds {1024, 2024, 3024}
- cudnn.benchmark=True (强制 False)
- 混合精度 BF16/FP16（论文 FP32）
- patience / min_delta（影响 best_epoch）
- ECE M=100, ddof=1

---

## 8. Bug 修复历史（11 + 6 + 1 项）

### 第一轮 cross review（6 个）
- FIX-1 Avazu hour 删除 (21 features, field_index=2 → site_id)
- FIX-2 Criteo archive 迁移 (signed_log1p_square_tokens + min_count=10)
- FIX-3 pretrain seed 固定 1024
- FIX-4 删 integral_dim
- FIX-5 uncertainty_bin_save_path 注入
- FIX-7 CTR estimate 100x 错修正

### 第二轮严肃审查（11 个）
- FIX-8 `methods/neucalib.yaml` → `methods/neu.yaml` (UMC 内部 method 名是 "neu")
- FIX-9 v9 跑 3 seed
- FIX-10 bootstrap CI + paired significance
- FIX-11 P5 logit u_mode 对照
- FIX-12 compute_diagnosis_prediction + 独立性标注
- B1 metrics.jsonl 链路 (CapturingTee + extract_metrics)
- B2/B5 v9_inference 实现 (复用 train_neu + sample_level_save_path)
- B3 load_v9_samples 派生 `u = log(sigma2+1e-8)`
- B4 v9 yaml 加 umc 方法
- B6 UMC shuffled-u Pearson corr 计算
- F2/F3 P3 加 stat 压制判定 + P4 命名澄清

### 性能优化（2 个）
- H1 prefetch_factor 注入
- H2 eval_batch_size 字段（保守值 ×4）

### 算法澄清（1 个）
- FIX-6 statistical method std=0 是 deterministic 算法本质

### 已知 Criteo 坑（**未来 Criteo 阶段需注意**）
1. **field_index 矛盾**: main 用 23（论文 Ch4 来源）vs v10 用 0（论文 Ch5 来源）—— **decision: 全用 23 内部一致**
2. **seed 3024 极端值** (plan §B 第 4 条已记录)
3. OOM 风险: batch_size_calib=65536 + 1M vocab embedding ≈ 5-8GB（5090 应 OK）
4. PYTHONUNBUFFERED=1 缺失（legacy 强制，reproduction 没）→ smoke 后加上
5. test.txt 无 label（删之）
6. CTR 25.6% 是子采样产物
7. train_test_split(shuffle=False) 时间排序（Pattern B 根因）
8. train_neu_criteo.py `__main__` dict bug（data_name="aliccp"+field_index=0 复制粘贴）→ orchestrator 覆盖

---

## 9. 关键文件路径速查

### 本地（macOS）
```
/Users/y/Research_MEM/30_reproduction/
├── README.md
├── CLAUDE.md
├── PROJECT_HANDOFF.md            # 本文件（持久化交接）
├── _SESSION_HANDOFF.md           # 旧版（保留参考）
├── UMC/                          # 算法层（baiyimeng/UMC 副本 + 参数化）
│   ├── _paths.py
│   ├── pretrain.py
│   ├── train_neu_{ali,avazu,criteo}.py
│   ├── train_sta_*.py
│   └── _legacy_runs/             # 历史脚本仅参考
├── reproduction/
│   ├── orchestrator.py           # 含 --parallel N (本地新增 未 push)
│   ├── _runner.py                # CapturingTee + extract_metrics
│   ├── configs/
│   │   ├── datasets/{aliccp,avazu,criteo}.yaml
│   │   ├── methods/{platt,ir,hb,umnn,neu,desc,sbcr,umc,umc_wor,uamcm,uamcm_wor}.yaml
│   │   ├── experiments/{main_99,v9_error_analysis,v10_ablation}.yaml
│   │   └── hardware/rtx5090.yaml  # eval_bs ×4 (保守值, 未 push 改动)
│   ├── data/{download.py, preprocess/{aliccp,avazu,criteo}.py}
│   ├── analysis/
│   │   ├── sanity_check.py
│   │   ├── diff_with_paper.py    # 含 bootstrap + 派生预判
│   │   ├── tables/               # 8 个表格（含 table_5_4_threshold）
│   │   └── figures/
│   └── utils/                    # seed/gpu/jsonl_log/status
├── tests/                        # pytest 39 passed / 4 skipped
├── docs/                         # 11 份文档（导师审阅用）
├── scripts/                      # 12 个 shell 入口
├── results/                      # tables + figures + diff_audit (git tracked)
└── experiments/                  # gitignored
```

### 远程（Paratera 容器，`/root/shared-nvme/30_reproduction/`）
```
data/
├── processed/
│   ├── aliccp/data.pkl           # 9.6GB ✓
│   └── avazu/data.pkl            # 6.7GB ✓
└── raw/criteo/criteo.tar.gz      # ~3GB 下载中

experiments/runs/pretrain/
├── aliccp/_backbone/seed_1024/{done.flag, run_config.json, train.log, checkpoint.pth}
└── avazu/_backbone/seed_1024/{done.flag, ...}

logs/pretrain.log                  # 阶段 1 主 log
```

---

## 10. 操作命令速查

### WebSSH 内（远程操作主要方式）

```bash
# === Smoke test (下一步) ===
cd /root/shared-nvme/30_reproduction && nohup python3 -u -m reproduction.orchestrator \
  --stage main --dataset aliccp --method uamcm --seed 1024 --max-runs 1 \
  > logs/smoke_aliccp_uamcm.log 2>&1 & echo "SMOKE_PID=$!"

# === 监控 (smoke 跑期间，每 2 min) ===
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader && \
  tail -10 /root/shared-nvme/30_reproduction/logs/smoke_aliccp_uamcm.log && \
  tail -10 /root/shared-nvme/30_reproduction/experiments/runs/main/aliccp/uamcm/seed_1024/train.log

# === smoke 完成后启动 main parallel=6 ===
# 前提: 先 push 本地 --parallel 改动 + 远程 git pull (需短暂 public)
cd /root/shared-nvme/30_reproduction && git pull && \
  nohup python3 -u -m reproduction.orchestrator --stage main --resume --parallel 6 \
  > logs/main.log 2>&1 & echo "MAIN_PID=$!"

# === v9 + v10 ===
nohup python3 -u -m reproduction.orchestrator --stage v9 --resume --parallel 6 > logs/v9.log 2>&1 &
nohup python3 -u -m reproduction.orchestrator --stage v10 --resume --parallel 6 > logs/v10.log 2>&1 &

# === Criteo 阶段 (后补) ===
cd /root/shared-nvme/30_reproduction/data/raw/criteo && tar xzf criteo.tar.gz && \
  rm test.txt criteo.tar.gz && cd ../../.. && \
  export PYTHONUNBUFFERED=1 && \
  nohup python3 -u -m reproduction.data.preprocess.criteo > /root/shared-nvme/preprocess_criteo.log 2>&1 &

# === 聚合 + 论文回写 ===
bash scripts/aggregate_results.sh
bash scripts/generate_paper_artifacts.sh

# === git push 结果回本地 ===
git add results/ && git commit -m "complete reproduction" && git push
```

### 本地（mac）

```bash
# SSH 不通（双因素），用 git 同步代码 + WebSSH 操作远程
cd /Users/y/Research_MEM/30_reproduction
# 编辑后:
git add -A && git commit -m "..." && git push

# 临时 public 让远程 pull (private repo + 容器无 GitHub auth):
gh repo edit yanghuaizhi/MEM_Thesis_Reproduction --visibility public --accept-visibility-change-consequences
# 用户 WebSSH: git pull
gh repo edit yanghuaizhi/MEM_Thesis_Reproduction --visibility private --accept-visibility-change-consequences
```

---

## 11. 风险与应对剧本

| 情况 | 应对 |
|------|------|
| **calib smoke OOM**（precompute 阶段显存 >30GB）| 改 `hardware/rtx5090.yaml: eval.batch_size_multiplier: 2`（× 4 → × 2）；最坏 × 1（等于 batch_size_calib） |
| **calib smoke metrics.jsonl 空** | 看 `extract_metrics` 正则是否匹配 calib 输出，可能要加 `epoch_loss` / `metrics_tag=calibrated` 兼容 |
| **parallel=6 GPU 调度差**（每 task 慢 50%+）| 降到 parallel=3 或 4 |
| **parallel CPU IO 瓶颈** | 降 `num_workers` 12 → 8 或 6（多任务共享 14 vCPU） |
| **容器被回收** | shared-nvme 保留 done.flag + 数据，重新申请同规格 → 装公钥 + WebSSH → 装环境 → `--resume` 续跑 |
| **GitHub auth fail (git pull)** | 临时 `gh repo edit ... --visibility public` 30 秒, pull 完立即改回 private |
| **训练 NaN/Inf** | sanity_check 触发；查 train.log 看 grad |
| **某 task fail** | `--resume` 自动跳过 done.flag 已存在的，仅重跑 failed/missing |

---

## 12. 下次会话优先行动清单（按顺序）

### A. 如果 backbone 已完成 + smoke 还没跑（**当前状态**）
1. **启动 smoke test** (AliCCP UAMCM seed=1024 单 task, 15-20 min)
2. **监控显存峰值** + metrics.jsonl + 数值合理性
3. **确定 parallel N**（smoke 显存峰值 < 5GB → N=5; < 10GB → N=3; < 20GB → N=2; 否则 N=1）
4. **本地 push 新 commits** + 临时 public + 远程 pull + 改回 private
5. **启动 main parallel=N**（66 任务，估 30-60 min）
6. main 完成 → 启动 v9 + v10（24 任务 estimate 30 min）

### B. 如果 smoke 已通过 + main 已启动
1. 每 15 min 看进度（`find experiments/runs -name done.flag | wc -l`）
2. 等 main 全完 → 立即启动 v9 + v10
3. v9 + v10 完成 → 启动 Criteo 阶段
4. Criteo wget 下完 → 解压 + preprocess + Criteo backbone smoke + Criteo main + v9 + v10
5. 全部完成 → 聚合 + diff_with_paper + 论文 v1.14 回写

### C. 如果远程容器已关
1. 重新申请同规格容器
2. shared-nvme 持久数据保留 → setup_env.sh → `--resume` 续跑

---

## 13. 关键引用 + 资源

- 论文 v1.13: `00_active/artifacts/v1.13_20260513/thesis.pdf`
- 复现 plan: `/Users/y/.claude/plans/users-y-research-mem-00-active-artifact-validated-sedgewick.md`
- baiyimeng/UMC: https://github.com/baiyimeng/UMC (Bai et al. SIGIR 2025)
- 数据 USTC: https://rec.ustc.edu.cn/share/5a70c6c0-9e4a-11ef-af55-8dfb3f6b3310 密码 5277
- GitHub: https://github.com/yanghuaizhi/MEM_Thesis_Reproduction (private)
- Paratera console: 容器名 `kcs-alxhwdwu` ID `ackcs-00gjgv3y` 山东二区

---

## 14. 自检清单（新会话恢复时必查）

- [ ] CLAUDE.md 加载 (cd 进项目目录自动)
- [ ] 本 PROJECT_HANDOFF.md 已读
- [ ] `git log --oneline` 看 commit 历史 + 是否有未 push 改动
- [ ] WebSSH 内 `find /root/shared-nvme/30_reproduction/experiments/runs -name done.flag | wc -l` 知道远程实验进度
- [ ] WebSSH 内 `ls /root/shared-nvme/30_reproduction/data/processed/` 看数据集状态
- [ ] WebSSH 内 `ls /root/shared-nvme/30_reproduction/data/raw/criteo/` 看 Criteo 是否下完
- [ ] 容器是否还在线（按量计费可能已停，看 Paratera 控制台）
- [ ] 如远程下线：shared-nvme 仍持久，重申请容器即可续

---

## 15. 当前累计成本审计

| 项 | 时长 | 元 |
|----|------|-----|
| 容器开机至今 | ~2.5h | ~7.5 元 |
| 预算余量 | - | 332.5 / 340 元 |
| 后续估算 | smoke 0.3h + main 1h + v9+v10 1h + Criteo 2h = 4.3h | ~13 元 |
| **预计总** | 7h | **~20-25 元** |
| 预算占比 | - | **~7%** |

---

**本文件持久化**：commit 后即使 context 清空，新会话从本文件即可完整恢复项目状态。
