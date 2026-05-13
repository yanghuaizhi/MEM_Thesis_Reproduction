# CLAUDE.md — 30_reproduction 项目指令

> 本文件在新会话 `cd` 进 30_reproduction/ 时自动加载，提供项目长期上下文。
> 会话状态见同目录 `_SESSION_HANDOFF.md`；完整规划见 `/Users/y/.claude/plans/users-y-research-mem-00-active-artifact-validated-sedgewick.md`。

## 项目定位

清华 MEM 学位论文《基于误差诊断的互联网广告点击率预估校准策略决策研究》（杨怀志, 导师张晨, v1.13 2026-05-13）的**完整实验复现项目**。

**目标**：以复现数据为依据更新论文 v1.14 全部实验数值，上传 GitHub `yanghuaizhi/MEM_Thesis_Reproduction` 交付导师审阅。

## 复现哲学（CRITICAL）

**verify（独立验证），不是 reproduce-to-match（数值匹配）。**

- 论文论断 = **假设**，不是 ground truth
- 复现独立运行，跑完后客观对照（支持/中立/反对）
- 复现与论断矛盾 → 先怀疑复现严谨性（避坑清单 + 配置详表），再考虑修订论文
- 论文 v1.13 数值是"先前实验的结果"，不是必须复现到的目标

详见 plan §A.4 + `_SESSION_HANDOFF.md` §3。

## 锁定的关键约束

| 项 | 值 |
|---|---|
| 颗粒度 | 11 方法 × 3 数据集 × 3 seeds |
| GPU | RTX 5090 单卡 32GB / 山东二区 / 2.98 元/h |
| 预算 | 340 元 ≈ 114h 总机时 |
| 数据集 | AliCCP, Avazu, Criteo（公开） |
| Seeds | {1024, 2024, 3024} |
| ECE bins M | **100**（不可改）|
| 标准差 ddof | **1**（Bessel 校正）|
| 模型选择 | **Loss-best**（按 LogLoss 选 epoch）|
| ECE 表述 | "ECE **降低 X%**"，禁用"改善 -X%" |
| cudnn.benchmark | **False**（保持严格可复现）|
| 混合精度 | **不用**（论文是 FP32）|
| v8 DA-SCL | **完全跳过**（已确认消融失败）|

## 项目结构

```
30_reproduction/
├── README.md                    项目首屏 + 一键复现
├── CLAUDE.md                    本文件
├── _SESSION_HANDOFF.md          会话交接（每次更新）
├── UMC/                         算法代码层（10_research_archive/UMC/ 副本 + 参数化）
│   ├── _paths.py                路径与依赖单点解析（new）
│   ├── calib/                   UMC + UAMCM 核心（不动）
│   ├── models/                  DeepFM + PackedDeepFM（不动）
│   ├── utils/                   metric.py + save_samples.py（不动）
│   ├── pretrain.py              已参数化路径
│   ├── train_neu_{ali,avazu,criteo}.py    已参数化
│   ├── train_sta_{ali,avazu,criteo}.py    已参数化
│   └── _legacy_runs/            v5-v10 历史脚本（仅参考）
├── reproduction/                复现工作层（新增）
│   ├── configs/                 YAML 配置中心
│   ├── orchestrator.py          统一编排器
│   ├── data/                    download + preprocess
│   ├── analysis/                tables + figures + sanity_check + diff_with_paper
│   └── utils/                   seed + logging + gpu + status
├── docs/                        10 份文档（导师重点看）
├── scripts/                     Shell 入口（11+ 脚本）
├── tests/                       单元 + smoke 测试
├── results/                     聚合产物（git 入库，含 markdown 表 + 图 + diff_audit）
├── data/                        数据（gitignored，存 /root/shared-nvme/）
└── experiments/                 运行时（gitignored）
```

## 关键路径

- 数据：`MEM_DATA_ROOT` env var > `30_reproduction/data/processed`
- Checkpoint: `MEM_CKPT_ROOT` env var > `30_reproduction/experiments`
- torch-uncertainty: `MEM_TORCH_UNCERTAINTY_SRC` env var > `10_research_archive/_archive/torch-uncertainty/src`

`UMC/_paths.py` 自动解析。验证命令：`python3 UMC/_paths.py`

## 与父目录的关系

- `/Users/y/Research_MEM/00_active/` — 论文修订（独立 git 仓库 `yanghuaizhi/MEM_Thesis`，v1.13 已发布）
- `/Users/y/Research_MEM/10_research_archive/` — 冻结研究归档（独立 git 仓库）
- `/Users/y/Research_MEM/30_reproduction/` — **本目录**，复现工作（独立 git 仓库 `yanghuaizhi/MEM_Thesis_Reproduction`，待 task #9 创建）

**禁止跨目录混合编辑**：
- 不要在 30_reproduction/ 编辑 00_active/thesis/*.md（论文回写经差异审计后再做）
- 不要在 30_reproduction/UMC/ 修改 10_research_archive/UMC/（前者是副本，后者是冻结归档）

## 工作流要点

### 本地工作（macOS + Claude）

1. 阅读 plan + HANDOFF 恢复上下文
2. 用 TaskList 跟踪 16 个任务（详见 HANDOFF §4）
3. 编辑代码、配置、文档
4. 本地 smoke test（pytest tests/）
5. git commit + push 到远程

### SSH 容器云工作（无 Claude）

1. `git clone yanghuaizhi/MEM_Thesis_Reproduction` 到 `/root/shared-nvme/`
2. `bash scripts/setup_env.sh`
3. `bash scripts/smoke_test_rtx5090.sh`
4. `bash scripts/download_data.sh` + `preprocess_data.sh`
5. 阶段 1-8 实验（按 RUNBOOK.md 操作）
6. 跑完 `git add results/ && git commit && git push`

### 本地诊断远程问题

```bash
# 看远程状态
ssh container 'cat /root/status.json' | jq .

# 实时日志
ssh container 'tail -f /root/shared-nvme/30_reproduction/logs/current.log'

# 拉元数据（小文件）
rsync -av --include="*.jsonl" --include="*.yaml" --exclude="*.pth" --exclude="*.npz" \
  container:/root/shared-nvme/30_reproduction/experiments/ ./experiments-meta/

# 远程出错时本地修复后 push，远程 pull + --resume
```

## 关键 don't

- 不要直接编辑 UMC/_legacy_runs/ 的脚本（仅参考）
- 不要改 UMC/calib/MonotonicNN.py（核心算法，改了数值漂移）
- 不要改 train batch_size / lr / epochs / seeds（破坏复现）
- 不要用 cudnn.benchmark = True（引入不确定性）
- 不要用混合精度（论文 FP32）
- 不要复现 v5/v6/v8 方法（已确认废弃）
- 不要假设论文当前数值是 ground truth（详见复现哲学）

## 资源限制

- 本地容量有限：数据集不下载到本地，仅在远程容器
- 大文件不入 git：`*.pth`, `*.npz`, `*.parquet`, `data/raw/`, `data/processed/`, `experiments/runs/`, `experiments/backbones/`
- pre-commit hook 自动拦截 >20MB 文件

## 引用

- 本项目：杨怀志, 张晨. (2026). 基于误差诊断的互联网广告点击率预估校准策略决策研究. 清华大学 MEM 学位论文.
- UMC 上游：Bai et al. (2025). *Unconstrained Monotonic Calibration*. SIGIR 2025. https://github.com/baiyimeng/UMC

---

**首次创建**：2026-05-13
**最后更新**：2026-05-13
