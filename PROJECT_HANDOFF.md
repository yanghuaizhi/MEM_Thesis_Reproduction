# PROJECT_HANDOFF.md — 30_reproduction 综合实施总结

> **本文件是项目持久化交接文档**。下次会话从这里恢复，能完整理解项目状态而不依赖历史对话。
> 最后更新：2026-05-14 16:00（**chain 5 阶段已 detach 在 tmux `mem` 内运行，约 25-30h 完成**）

---

## 0. 下次会话开场提示词（复制粘贴）

```
继续推进 /Users/y/Research_MEM/30_reproduction/ 论文复现项目。

请先 cd 到 /Users/y/Research_MEM/30_reproduction/，让项目 CLAUDE.md 加载，然后阅读
PROJECT_HANDOFF.md（本文件）。

当前状态（2026-05-14 16:00 启动）：
- 远程 tmux session `mem` 内跑 5-stage chain (backbone → main smoke → main → v9 → v10)
- chain_run.sh PID 在 /tmp/chain_run.sh，主 log 在 logs/chain_master.log
- SSH **完全可用**（密码 + publickey 双因素已通过，mux master 在跑）
- 三个修复 commit 已 push（1cf8bb4 / 896cff3 / 0e23329），远程已拉

第一件事：检查 chain 进度
  ssh paratera 'tail -30 /root/shared-nvme/30_reproduction/logs/chain_master.log'
  ssh paratera 'find /root/shared-nvme/30_reproduction/experiments/runs -name done.flag | wc -l'

根据进度判断：
  A. 还在跑 → 监控 + 解答问题，不要打断
  B. 全部 done (chain_completed.flag 存在) → 进入 aggregate 阶段
  C. 中间失败 → 看 logs/chain_master.log 末尾 "[chain] STAGE X FAILED"，--resume 重启
```

---

## 0.5 当前状态速读（2026-05-14 16:00）

### Chain 运行中
- tmux session：**`mem`**（远程 `tmux attach -t mem` 可看）
- launcher 脚本：`/tmp/chain_run.sh`（远程 tmpfs，容器重启会丢；可重新构造，逻辑见 §13.5）
- 主 log：`/root/shared-nvme/30_reproduction/logs/chain_master.log`
- 5 个 stage chain 用 `&&` 串联，单点失败即停

### Stage 进度（用 done.flag 计数）
```
ssh paratera 'echo pretrain:$(find .../runs/pretrain -name done.flag|wc -l)/3 ...'
```
| Stage | 任务 | 进度 |
|------|------|------|
| 1 backbone | 3 | 已 done 2/3 (aliccp+avazu), criteo 在 Epoch 7+ 训练中 |
| 2 main smoke | 2 | 0/2 (待 backbone 完) |
| 3 main 剩余 | 97 | 0/97 |
| 4 v9 | 12 | 0/12 |
| 5 v10 | 18 | 0/18 |

### 完成里程碑（参考时刻）
| 时刻 | 应看到 |
|------|--------|
| 16:21 ± | Criteo backbone done → STAGE 2 启动 |
| 16:56 ± | main smoke 2 done → 关键防线通过 |
| 次日 16:30 ± | chain 全部 done，`experiments/chain_completed.flag` 写入 |

### 容错保证
- 进度持久化到 `shared-nvme/.../done.flag`，**容器重启不丢**
- 单 task 失败 ≤ 30 min 损失（其他 task 不受影响，`--resume` 接力）
- tmux server 挂 → 跑完的 done.flag 仍在，重启 chain 用 `--resume` 跳过已 done

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

### 3.2 远程容器进度（2026-05-14 16:00 chain 启动时刻）

| 阶段 | 状态 | 实测/估时 | 关键产出 |
|------|------|---------|---------|
| 0 环境配置 | ✓ 完成 | 30min | torch-uncertainty + lightning + pytest 39 passed |
| 0.5 数据 AliCCP+Avazu | ✓ 完成 | 25min | `data/processed/aliccp/data.pkl` 9.6GB + `avazu/data.pkl` 6.7GB |
| 0.6 Criteo 数据 | ✓ **完成 16:00 前** | 25min preprocess（chunked 两遍扫）| `data/processed/criteo/data.pkl` 6.9GB (45.8M 行, CTR 0.2562) |
| 1 backbone aliccp+avazu | ✓ 完成 | 22min | 2 个 done.flag |
| 1' **backbone criteo** | 🟡 **chain Stage 1 进行中** (16:00 启动) | 估 25 min, **Epoch 7+/20 16:03** | 待 done.flag, val_auc 0.8068 单调上升 |
| 1.5 smoke uamcm/aliccp | ✓ 完成 (单 task 30.4min) | val ECE 降 55% 合理 | 1/99 main done |
| 2 main smoke 2 task | 📋 chain Stage 2 待启 | 估 35 min (parallel=2) | 验证 num_workers=6 + parallel=2 新配置 |
| 3 main 剩余 97 | 📋 chain Stage 3 待启 | **估 ~20h** (parallel=2) | 99/99 calib metrics.jsonl |
| 4 v9 sample-level | 📋 chain Stage 4 | 估 ~1.5h | 12 NPZ |
| 5 v10 ablation | 📋 chain Stage 5 | 估 ~3h | 18 u_mode 消融 |
| 6 aggregate + diff | 📋 chain 完后人工 | <30min CPU | results/{tables,figures,diff_audit}/ |

### 3.3 估时（基于实测数据，非乐观估算）

实测依据：
- AliCCP backbone 17 min, Avazu 5 min（real, num_workers=12）
- smoke `main/aliccp/uamcm/seed_1024` **30.4 min**（real, single task）

| 阶段 | 任务 | 单 task | 并行 | 估时 |
|------|------|--------|------|------|
| 1 Criteo backbone | 1 | 25 min | 1 | 25 min |
| 2 main smoke | 2 | 30 min | 2 | 35 min |
| 3a main statistical 余下 | 24 | ~8 min | 2 | 1.6h |
| 3b main neural 余下 | 71 | ~30 min | 2 | ~18h |
| 4 v9 | 9 (3 done) | ~20 min | 2 | 1.5h |
| 5 v10 | 18 | ~20 min | 2 | 3h |
| 6 aggregate | -- | -- | 1 | <30 min |
| **合计** | | | | **保守 25-30h, 乐观 20-22h** |

预算：~80-95 元 / 340 元，**3-4x 缓冲**。

---

## 4. 远程容器信息

| 项 | 值 |
|----|------|
| 平台 | Paratera 容器云（按量计费 2.98 元/h） |
| 区域 | 山东二区 |
| GPU | RTX 5090 32GB (sm_120 Blackwell) |
| **CPU quota (cgroup v2)** | **14 vCPU 等价**（`cpu.max=1400000 100000`；`nproc=128` 是 host 不可信）|
| **RAM (cgroup memory.max)** | **100 GB**（`free -h` 显示 1Ti 是 host 不可信）|
| /dev/shm | 50 GB tmpfs |
| OS / PyTorch | Ubuntu 24.04 + PyTorch 2.7 (sm_120 支持) |
| CUDA | 12.8 |
| 共享存储 | 80GB shared-nvme，剩余 ~37GB（容器回收不丢，vast 网络挂载）|
| SSH 入口 | `ssh.bj8.bz1.paratera.com:2233` user=`root@ackcs-00gjgv3y` |
| 服务端 | **SSHPiper** 反向代理网关（多租户分流，按用户名后缀路由）|
| 认证 | **MFA：publickey + password** 双因素（publickey "partial success" 后必须 password）|
| **本地 SSH alias** | `paratera`（~/.ssh/config 已配，**实测可用**）|
| **SSH 复用** | ControlMaster mux 已启用，detach 不影响后续 `ssh paratera ...` |
| **远程 tmux** | 已 `apt-get install tmux 3.4`，当前 session：`mem` |
| 工作目录 | `/root/shared-nvme/30_reproduction/` |
| 数据 | `data/processed/{aliccp,avazu,criteo}/data.pkl` 全 ✓ |
| 实验目录 | `experiments/runs/<stage>/<dataset>/<method>/seed_<N>/` |
| Log 目录 | `logs/`（`chain_master.log` 是当前主 log）|

### 4.1 容器回收应对

shared-nvme 持久。重新申请同规格容器后：
1. SSH 直接可用（不需要再做认证设置）
2. data + experiments + done.flag 全保留
3. `cd /root/shared-nvme/30_reproduction && pip install torch-uncertainty lightning`
4. 重建 `/tmp/chain_run.sh`（脚本逻辑见 §13.5）并 `tmux new -s mem` 启动
5. `--resume` 跳过已 done 的 task

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

## 6. 当前任务状态（2026-05-14 16:00）

### ✓ 已完成
- Stage 0 远程环境配置 + AliCCP/Avazu data + 2 backbone
- Stage 0.6 Criteo preprocess（chunked 两遍扫修复挂死，data.pkl 6.9GB ✓）
- Stage 1.5 smoke `main/aliccp/uamcm/seed_1024`（30.4min, ECE 降 55%）
- 三个关键 commit push + 远程 pull
- tmux 装好

### 🟡 进行中（detached 自动跑）
- **chain 5 stage** 在 tmux `mem`，主 log `logs/chain_master.log`
  - Stage 1 Criteo backbone（Epoch 7+/20，预计 16:21 完成）
  - Stage 2 main smoke 2 task（关键防线，验证 parallel=2 + num_workers=6）
  - Stage 3 main 剩余 97
  - Stage 4 v9 12 task
  - Stage 5 v10 18 task

### 📋 chain 完后的人工动作
1. `bash scripts/aggregate_results.sh` (sanity + tables + figures)
2. `bash scripts/generate_paper_artifacts.sh` (4 layer diff_audit)
3. 同步 `results/` 回本地 + git commit
4. 看 `results/diff_audit/diff_with_v1_13.md` 决定 v1.14 修订点
5. 在 `00_active/thesis/*.md` 修订 + `make pdf` 出 v1.14 交导师

---

## 7. 智能加速层级（**已落地最终配置**）

### Tier 1 已开（数值 0 影响）
- `num_workers` 分场景：
  - **`num_workers: 6`** (main/v9/v10, parallel=2 时 2×6+2=14 vCPU)
  - **`num_workers_pretrain: 12`** (pretrain 单任务, 12+1=13 vCPU)
- `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4` (H1)
- `eval_batch_size=calib×4` (保守值, 待 smoke 后可上调 ×8)

### Tier 2 已开（理论 1e-7 影响）
- TF32 matmul + cudnn ✓
- matmul_precision=high ✓
- torch.compile: **关**（Blackwell + PT 2.7 兼容性未验证）

### --parallel 度（**已固化默认值**）
- pretrain: **1**（单任务，scripts/run_pretrain.sh 默认）
- main/v9/v10: **2**（scripts/run_*.sh 默认 `--parallel 2`，可 CLI 覆盖）
- ⚠️ **禁止 ≥3**：实测 2026-05-14 14:36 parallel=4 撞 cgroup 资源（CPU/RAM/shared mem 不明确）导致 4 task 同时挂死

### Tier 3 红线（禁止改）
- train batch_size, lr, num_estimators=16, dropout, init_std, l2_reg, embedding_dim, hidden_units
- seeds {1024, 2024, 3024}
- cudnn.benchmark=True (强制 False), cudnn.deterministic=True
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

### 本地 macOS → 远程 SSH 直连（**当前主要方式**）

```bash
# === 看 chain 总进度 ===
ssh paratera 'tail -30 /root/shared-nvme/30_reproduction/logs/chain_master.log'

# === 看各 stage done.flag 计数 ===
ssh paratera 'cd /root/shared-nvme/30_reproduction && \
  echo "pretrain: $(find experiments/runs/pretrain -name done.flag|wc -l)/3"; \
  echo "main    : $(find experiments/runs/main -name done.flag 2>/dev/null|wc -l)/99"; \
  echo "v9      : $(find experiments/runs/v9 -name done.flag 2>/dev/null|wc -l)/12"; \
  echo "v10     : $(find experiments/runs/v10 -name done.flag 2>/dev/null|wc -l)/18"'

# === 看 GPU / load / 内存 ===
ssh paratera 'nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader; uptime; free -h | head -2'

# === attach tmux 看实时 stdout (Ctrl-B D 退出 attach) ===
ssh -t paratera 'tmux attach -t mem'

# === 看当前正在跑的 task 的 train.log 末尾 ===
ssh paratera 'find /root/shared-nvme/30_reproduction/experiments/runs -name train.log -newer /tmp/chain_run.sh | xargs -I{} tail -5 {}'

# === chain 完成后的人工动作 ===
ssh paratera 'cd /root/shared-nvme/30_reproduction && bash scripts/aggregate_results.sh && bash scripts/generate_paper_artifacts.sh && git add results/ && git commit -m "results: full chain produced" && git push'
git pull            # 本地同步 results/
open results/diff_audit/diff_with_v1_13.md
```

### 远程 ad-hoc 操作

```bash
# === 杀掉 chain（用户主动停） ===
ssh paratera 'tmux kill-session -t mem'

# === 单独跑某阶段（chain 失败后用 --resume 接力） ===
ssh -t paratera 'tmux new -A -s mem'
# tmux 内:
cd /root/shared-nvme/30_reproduction && bash scripts/run_main_experiments.sh --resume

# === 查单个失败 task 的错误 ===
ssh paratera 'find /root/shared-nvme/30_reproduction/experiments/runs -name error.flag -exec cat {} \;'
```

---

## 11. 风险与应对剧本

| 情况 | 应对 |
|------|------|
| **chain Stage 2 main smoke 又挂死** | parallel=2 不够保守，降到 1 + num_workers=8。改 `scripts/run_main_experiments.sh` `PARALLEL_DEFAULT=2 → 1`。这是关键防线，挂了就停！别盲目重试 |
| **某个 main task 出 NaN/Inf** | sanity_check 会标记。其他 task 不受影响，--resume 跳过 done 的，重跑出错的 |
| **GPU OOM**（main calib 显存 >30GB）| 改 `hardware/rtx5090.yaml: eval.batch_size_multiplier: 4 → 2 或 1`。push + 远程 pull + `--resume` |
| **vast 网络存储慢/不可用** | 等几分钟。如长时间不通，看 paratera 平台状态。data.pkl 已 load 到 RAM 不依赖 IO |
| **容器被回收** | shared-nvme 保留 done.flag + 数据。重新申请同规格 + git clone + setup_env + 重建 /tmp/chain_run.sh + tmux new + chain --resume 续跑 |
| **训练 NaN/Inf** | sanity_check 触发；查 train.log 看 grad |
| **某 task fail** | `--resume` 自动跳过 done.flag 已存在的，仅重跑 failed/missing |

---

## 12. 下次会话优先行动清单（按顺序）

### 第一件事：检查 chain 状态

```bash
ssh paratera 'tail -30 /root/shared-nvme/30_reproduction/logs/chain_master.log'
ssh paratera 'cd /root/shared-nvme/30_reproduction && \
  echo pretrain:$(find experiments/runs/pretrain -name done.flag|wc -l)/3 \
  main:$(find experiments/runs/main -name done.flag 2>/dev/null|wc -l)/99 \
  v9:$(find experiments/runs/v9 -name done.flag 2>/dev/null|wc -l)/12 \
  v10:$(find experiments/runs/v10 -name done.flag 2>/dev/null|wc -l)/18'
ssh paratera 'test -f /root/shared-nvme/30_reproduction/experiments/chain_completed.flag && echo CHAIN_DONE || echo CHAIN_RUNNING_OR_FAILED'
```

### 按状态分支

**A. chain 还在跑**（log 末尾无 STAGE FAILED + 无 chain_completed.flag）
- 监控不打断，回答用户问题。chain 自己跑完。

**B. chain 完成**（看到 `experiments/chain_completed.flag`）
- 跑 aggregate + paper_artifacts（CPU bound, <30min）：
  ```bash
  ssh paratera 'cd /root/shared-nvme/30_reproduction && bash scripts/aggregate_results.sh && bash scripts/generate_paper_artifacts.sh && git add results/ && git commit -m "results: chain done $(date +%Y%m%d)" && git push'
  git pull
  ```
- 让用户看 `results/diff_audit/diff_with_v1_13.md`，决定 v1.14 修订点

**C. chain 中间失败**（log 末尾有 STAGE X FAILED）
- 根据 STAGE 号定位：
  - STAGE 1/2: 配置问题，看 train.log 末尾，可能 OOM 或 dataloader 问题
  - STAGE 3+: 个别 task 可能崩，多数 done.flag 已存在，--resume 接力
- 修后在 tmux 内重启 chain 或单独跑剩余 stage

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

## 13. 自检清单（新会话恢复时必查）

- [ ] CLAUDE.md 加载 (cd 进项目目录自动)
- [ ] 本 PROJECT_HANDOFF.md 已读
- [ ] `git log --oneline` 看 commit 历史 + 是否有未 push 改动
- [ ] `ssh paratera 'tail -30 /root/shared-nvme/30_reproduction/logs/chain_master.log'` 看 chain 状态
- [ ] `ssh paratera 'find /root/shared-nvme/30_reproduction/experiments/runs -name done.flag | wc -l'` 知道远程实验进度
- [ ] `ssh paratera 'test -f /root/shared-nvme/30_reproduction/experiments/chain_completed.flag && echo DONE || echo RUNNING'`
- [ ] 容器是否还在线（按量计费可能已停，看 Paratera 控制台）
- [ ] 如远程下线：shared-nvme 仍持久，重申请容器后重建 /tmp/chain_run.sh（见 §13.5）并 `--resume` 续跑

---

## 14. 当前累计成本审计（2026-05-14 16:00）

| 项 | 时长 | 元 |
|----|------|-----|
| 容器开机至今 | ~5.5h | ~16 元 |
| chain 预估 25-30h | ~28h | ~83 元 |
| aggregate + diff | ~0.5h | ~1.5 元 |
| **预计总** | ~34h | **~100 元** |
| 预算占比 | - | **~30% / 340 元** |
| 余量 | - | 240 元（3 个数据集全跑后仍剩 ~70%） |

---

## 15. 今日进展时间线（2026-05-14）

### 关键事件
| 时刻 | 事件 |
|------|------|
| 11:50 | smoke `main/aliccp/uamcm/seed_1024` 完成（30.4min, ECE 0.0173 → 0.0078, 降 55%）|
| 14:12 | Criteo preprocess 启动后挂死（pandas 一次性读 11G TSV, 24min 无 stdout）|
| 14:36 | main `--parallel 4` 启动，4 task 全在"Feature names"后挂死 |
| 15:13 | 新 session 启动诊断：SSH 通畅、确认 cgroup 真实硬件 |
| 15:25 | commit `1cf8bb4`：Criteo preprocess chunked 两遍扫修复 |
| 15:26 | 远程跑 chunked preprocess（PID 22933 替代挂死的 14:12 进程）|
| 15:34 | commit `896cff3`：main/v9/v10 num_workers 12→6 + scripts 默认 parallel=2 |
| 15:41 | Criteo preprocess 完成（6.9GB data.pkl, CTR 0.2562 一致）|
| 15:44 | commit `0e23329`：pretrain 用 num_workers_pretrain=12（分场景 num_workers）|
| 15:49 | 远程装 tmux 3.4 |
| 15:54 | git checkout 恢复 main_99/v9/v10 yaml 含 criteo |
| 15:56 | chain 5 stage 在 tmux `mem` 启动 |
| 16:00 | Criteo backbone Epoch 7+/20，val_auc 0.8068 单调上升 |

### 三个新 commit（已 push）
```
0e23329 opt: pretrain uses num_workers_pretrain=12 (main keeps 6)
896cff3 fix: main/v9/v10 CPU starvation (parallel=4 × num_workers=12 = 52 process on 14 vCPU)
1cf8bb4 fix: chunked Criteo preprocess (11G TSV hang on vast network storage)
```

### 重要事实修正（在 §4 已更新）
- ~~"SSH 双因素 auth Claude 无法用，靠 WebSSH"~~ → SSH 完全可用（publickey + password mux master）
- ~~"CPU/RAM 14vCPU/120GB"~~ → cgroup v2 limit: **14 vCPU quota / 100 GB RAM**（`nproc=128` 和 `free=1Ti` 是 host 不可信）
- ~~"parallel=6 估 ~3h"~~ → parallel=4 已被证伪挂死；parallel=2 估 main **~20h**

### 经验教训
1. **vast 网络存储 + 一次性大 IO 致 stdout 假死**：chunked + flush=True 解决
2. **cgroup v2 OOM 不入 dmesg**：之前 parallel=4 挂死无 OOM 日志证据，但 RAM cgroup 100GB 估算 4 task × 8G + 48 worker copy 确有撞墙嫌疑
3. **`nproc` / `free -h` 在容器内不反映 cgroup 限制**：必查 `cpu.max` 和 `memory.max`
4. **PyTorch DataLoader fork + persistent_workers + parallel 多进程是组合炸弹**：单进程 num_workers=12 OK，4 parallel × num_workers=12 = 死
5. **MFA 服务端（SSHPiper）会反复跟客户端打 "partial success" 协议**：BatchMode=yes 会让 publickey-only 卡住，需要 password 兜底

---

## 16. /tmp/chain_run.sh 完整脚本（容器重启后重建用）

`/tmp` 在容器系统盘，重启会丢。重建用此脚本：

```bash
#!/bin/bash
cd /root/shared-nvme/30_reproduction
set -o pipefail
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "===== [$(ts)] STAGE 1/5: Criteo backbone pretrain (single task) ====="
bash scripts/run_pretrain.sh --resume 2>&1 | tee logs/chain_pretrain.log
RC=$?
[ $RC -ne 0 ] && { echo "[chain] STAGE 1 FAILED (rc=$RC), STOP"; exit $RC; }

echo
echo "===== [$(ts)] STAGE 2/5: main SMOKE verify (parallel=2, max-runs=2) ====="
python3 -m reproduction.orchestrator --stage main --resume --parallel 2 --max-runs 2 2>&1 | tee logs/chain_main_smoke.log
RC=$?
[ $RC -ne 0 ] && { echo "[chain] STAGE 2 FAILED (rc=$RC), STOP — check parallel=2 config"; exit $RC; }

echo
echo "===== [$(ts)] STAGE 3/5: main 99 (resume from 2 done) ====="
bash scripts/run_main_experiments.sh 2>&1 | tee logs/chain_main_full.log
RC=$?
[ $RC -ne 0 ] && { echo "[chain] STAGE 3 FAILED (rc=$RC), STOP"; exit $RC; }

echo
echo "===== [$(ts)] STAGE 4/5: v9 sample-level inference ====="
bash scripts/run_v9_error_analysis.sh 2>&1 | tee logs/chain_v9.log
RC=$?
[ $RC -ne 0 ] && { echo "[chain] STAGE 4 FAILED (rc=$RC)"; exit $RC; }

echo
echo "===== [$(ts)] STAGE 5/5: v10 u_mode ablation ====="
bash scripts/run_v10_ablation.sh 2>&1 | tee logs/chain_v10.log
RC=$?
[ $RC -ne 0 ] && { echo "[chain] STAGE 5 FAILED (rc=$RC)"; exit $RC; }

echo
echo "===== [$(ts)] ALL STAGES COMPLETE ====="
date > experiments/chain_completed.flag
```

**启动**：
```bash
ssh -t paratera 'tmux new -A -s mem' 进入 tmux
chmod +x /tmp/chain_run.sh
/tmp/chain_run.sh 2>&1 | tee logs/chain_master.log
# Ctrl-B 然后 D detach
```

---

## 17. 关键引用 + 资源

- 论文 v1.13: `00_active/artifacts/v1.13_20260513/thesis.pdf`
- 复现 plan: `/Users/y/.claude/plans/users-y-research-mem-00-active-artifact-validated-sedgewick.md`
- baiyimeng/UMC: https://github.com/baiyimeng/UMC (Bai et al. SIGIR 2025)
- 数据 USTC: https://rec.ustc.edu.cn/share/5a70c6c0-9e4a-11ef-af55-8dfb3f6b3310 密码 5277
- GitHub: https://github.com/yanghuaizhi/MEM_Thesis_Reproduction (private)
- Paratera console: 容器名 `kcs-alxhwdwu` ID `ackcs-00gjgv3y` 山东二区

---

**本文件持久化**：commit 后即使 context 清空，新会话从本文件即可完整恢复项目状态。
