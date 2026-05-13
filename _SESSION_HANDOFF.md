# _SESSION_HANDOFF.md - 30_reproduction 复现项目会话交接

> 本文件是会话交接文档。下次会话从这里开始，能完整恢复到当前状态而不丢任何上下文。
> 最后更新：2026-05-14（#15 完成）

---

## 0. 下一会话的开场提示词（直接复制到 Claude）

```
继续推进 /Users/y/Research_MEM/30_reproduction/ 复现项目。

请先 cd 到 /Users/y/Research_MEM/30_reproduction/ 让项目 CLAUDE.md 加载，然后阅读 _SESSION_HANDOFF.md（本文件）了解会话交接，再阅读 plan 文件 /Users/y/.claude/plans/users-y-research-mem-00-active-artifact-validated-sedgewick.md 了解完整规划。

当前进度：16 个任务完成 4 个（#13 骨架、#1 cp UMC、#6 参数化路径、#15 reproduction/utils/），需要继续 #16 reproduction/configs/ 全部 YAML。请用 TaskList 工具重建任务列表，标记 #13/#1/#6/#15 为 completed，然后开始 #16。
```

---

## 1. 项目目标与定位（一句话）

为清华 MEM 学位论文《基于误差诊断的互联网广告点击率预估校准策略决策研究》（v1.13, 2026-05-13）建立独立、自动化、文档完备的实验复现项目，**以复现数据为依据**更新论文 v1.14，上传 GitHub `yanghuaizhi/MEM_Thesis_Reproduction` 供导师审阅。

**论文核心论断**（要复现验证的）：
- 三种误差模式可区分（AliCCP 模式 A 强过预测 / Criteo 模式 B 弱欠预测 / Avazu 模式 C 非单调混合）
- UAMCM 在三个数据集差异化表现（AliCCP -13.7%, Avazu -17.6%, Criteo -46.9%）
- shuffled-u 消融印证 u 信号的场景依赖性（AliCCP/Criteo +70% / +69% 恶化，Avazu -7.9% 落在 σ 内）
- 决策框架 S1=Criteo, S2=AliCCP, S3=Avazu 三场景对应保持
- **诊断预判 = 实验验证**（论文最关键论证支点）

---

## 2. 已锁定的决策（不需要再讨论）

| 决策项 | 选定值 | 来源 |
|--------|-------|------|
| 项目位置 | `/Users/y/Research_MEM/30_reproduction/`（与 00_active/、10_research_archive/ 平行）| 用户确认 |
| 复现颗粒度 | **完整 11 方法 × 3 数据集 × 3 seeds** | 用户确认 |
| GPU 配置 | RTX 5090 单卡 32GB / 14vCPU / 120GB RAM / 山东二区 / 2.98 元/h | 用户确认（已纠正双卡显存误解）|
| 预算 | 340 元 ≈ 114h 总机时 | 用户确认 |
| RTX 5090 性能假设 | 基础 1.3x / Tier 1 1.35-1.4x / +TF32 1.5-1.7x / +compile 1.6-1.9x | architect agent 评估 |
| v9 sample-level 数据 | 完整重新生成（不复用历史 2GB）| 用户确认 |
| 数据下载策略 | aria2c + md5 + retry 自动化脚本 | 用户确认 |
| 数据存储位置 | SSH `/root/shared-nvme/`，扩容到 80GB | 用户确认 |
| UMC/ 修改边界 | **不设硬边界**，必要时改训练逻辑（用户解锁）| 用户确认 |
| 项目结构 | minimal-change（UMC/ 仅参数化路径 + Tier 优化挂钩；reproduction/ 新增层）| 用户确认 |
| 导师审阅 | **不审阅代码**，只看 docs/ + results/ 总结 | 用户确认 |
| 复现哲学 | **verify（独立验证）**，不是 reproduce-to-match（数值匹配） | 用户确认（多轮修正后的核心原则）|

---

## 3. 复现哲学：verify, not reproduce-to-match（CRITICAL，必读）

**核心原则**（推翻了早期"数值匹配"的思路）：

| 错误的复现哲学 | 正确的复现哲学（本项目采用） |
|--------------|-------------------------|
| "跑出与论文一致的数值才算成功" | "用严谨的实验流程独立生成数据，然后客观评估是否支撑论文论断" |
| 论文论断 = 真理；复现 = 验证手段 | 论文论断 = 假设；复现 = 独立证据 |
| 复现失败 = 我们错了 | 复现与论断矛盾 = 至少一方需要修订（先确认复现严谨，再判断论文是否需要修订）|
| ECE 差 > 0.5% 就报警 | ECE 数值在合理范围；论断方向对得上即可 |

**判定逻辑**（详见 plan 文件 §A.4）：
1. 复现独立运行，不预设论文论断为真
2. 先做内部质量门（M.0）：训练正常、指标合理、ddof=1、M=100、shuffled-u 真打乱
3. 通过 M.0 后做四层验证：L1 诊断（P1）/ L2 方法（P2-P4）/ L3 机制（P5）/ L4 决策（S1-S3）
4. 三态评估每个论断：支持 / 中立 / 反对
5. 多数反对 → **优先怀疑复现配置错**（先查避坑清单），逐项排除后才考虑修订论文

**论文数据可靠性自审**（plan §A.5）：
- v9 NPZ 可信度高（Ch3 PCOC），但复现仍独立生成
- Ch4 主结果 CSV 可信度**中-低**（v5 Avazu 数据对齐 bug 历史发生过），复现后**优先信任复现数据**
- v10 shuffled-u 实现要**独立审计代码**（用 Pearson |corr| < 0.01 验证打乱）

---

## 4. 当前实现状态（2026-05-14）

### 4.1 已完成（4/16 任务）

| Task | 状态 | 产物 | 行数 |
|------|------|------|-----|
| #13 创建项目骨架 | ✓ | 24 子目录 + README.md (4553B) + LICENSE + .gitignore (1854B) + .githooks/pre-commit (executable) + pyproject.toml + environment.yml + data/README.md | ~250 行 |
| #1 cp UMC 副本 + _legacy_runs 整理 | ✓ | UMC/ 副本（1.9MB，rsync 排除 .ipynb_checkpoints）+ 9 个历史 run_v*.py 移入 _legacy_runs/ + README_UMC.md (124 行) + _legacy_runs/README.md (87 行) | ~211 行新增 |
| #6 参数化 UMC 训练入口 | ✓ | 新建 `UMC/_paths.py` (131 行) + 修改 7 个文件 (pretrain + 3 train_neu_* + 3 train_sta_*) 路径硬编码 | 131 + ~80 行 diff |
| #15 reproduction/utils/ 工具模块 | ✓ | `reproduction/__init__.py` + `reproduction/utils/{__init__,seed,gpu,jsonl_log,status}.py`，5 文件 587 行；4 模块均带 `if __name__ == "__main__"` self-check 并通过 | 587 行 |

**总变更**：约 2,590 行新增 + ~80 行 diff。

### 4.2 进行中

无（task #15 刚完成）。

### 4.3 待办（12/16 任务，按依赖顺序）

| Task | 估时 | 依赖 | 必读规范 |
|------|------|------|---------|
| #16 reproduction/configs/ 全部 YAML | 60-90 min | #15 | plan §C（配置详表）、§B（datasets/methods/experiments）|
| #12 reproduction/orchestrator.py 编排器 | 1.5-2h | #15, #16 | plan §E.2/§E.3（subprocess 调用 UMC/train_*.py + done.flag + --resume）|
| #14 reproduction/data/ 下载与预处理 | 60-90 min | #15, #16 | plan §H 阶段 0、data/README.md（USTC + Kaggle 双源）|
| #11 reproduction/analysis/sanity_check.py | 60-90 min | #15 | plan §A.6.1（M.0 内部质量门，不依赖论文数值）|
| #2 reproduction/analysis/diff_with_paper.py | 2-3h | #11 | plan §M.2-M.6（四层验证 L1/L2/L3/L4 + 三态评估）|
| #10 reproduction/analysis/tables/ | 60-90 min | #11 | plan §A.2（每个表的来源映射）|
| #3 reproduction/analysis/figures/ | 60-90 min | #11 | plan Ch3 图 3-1~3-4 + Ch4 图 4-1~4-2 |
| #4 scripts/ Shell 入口（12 个）| 90 min | #12, #14 | plan §E.3（11 个 + setup_env + smoke_test）|
| #5 tests/ 单元 + smoke 测试 | 60-90 min | #15 | plan §M.1（参数化前后输出 1e-6 容差）+ TF32 drift |
| #8 docs/ 文档（10 份）| 2-3h | 全部完成后 | plan §F、§G、§J、§H |
| #7 本地 smoke test + 修 bug | 30-60 min | 全部完成 | plan §M.7 |
| #9 git init + push | 30 min | #7 | plan §K.1 |

**剩余估时合计**：约 13-20 小时纯编码 + 2-3 小时调试 = **3-4 天**。

---

## 5. 重要的非显然发现（本次会话学到的，未在 plan 文件展开）

### 5.1 RTX 5090 性能初估的过乐观（已纠正）

- architect agent 给的 2.2x 是基于 CUDA cores + 带宽的**理论**估算
- 实际 DeepFM 类小模型加速比 1.3-1.6x（kernel overhead 主导）
- 已在 plan §A 中修正为 1.5x 基准

### 5.2 RTX 3090 双卡的显存认知（已纠正）

- "2 × 24GB" **不等于 48GB 大显存**——两卡独立各 24GB
- 单任务 OOM 风险与单卡 24GB 完全相同
- 双卡仅在 DDP/Model Parallel 时合并显存，现有 UMC 代码无此实现
- 最终选 A1 RTX 5090（32GB 真显存，零 OOM 风险）

### 5.3 项目结构演进（minimal-change 战胜重组方案）

经历三轮讨论：
1. 最初方案：完全重组 UMC 代码为 `src/repro_ctr/` 包结构（被否，工程量大且数值漂移风险高）
2. 次轮方案：UMC/ 子目录 + reproduction/ 子目录二层（被部分接受）
3. **最终方案**：UMC/ 修改边界放开但仍 minimal-change（仅参数化路径 + Tier 优化挂钩，~150 行 diff）

### 5.4 train_neu_criteo.py 的已知 bug（保留不修复）

- L1002 `"data_name": "aliccp"` 与文件名不符（应为 "criteo"）
- L1014 `"field_index": 0` 应为 23
- L1028 `"phase4_uncertainty_bins_aliccp.csv"` 应为 criteo
- **保留原状**：orchestrator 通过 YAML 传入正确的 data_name + field_index 覆盖；不动 UMC 源减少数值漂移

### 5.5 paper_update remote 与 origin 的差异

- `10_research_archive` git 有两个 remote：`origin`（yanghuaizhi/PAPER）+ `paper_update`（yanghuaizhi/PAPER_Update）
- paper_update 仅多 1 个 commit `b47a40d`（MEM thesis backup），与 origin 内容基本一致
- **可忽略**，复现工作基于本地副本即可

### 5.6 v8 完全跳过的根因

- v8_validation 的 UAMCM_no_u_rs + DA-SCL 已被 architect 调研确认**消融失败**（不收敛或恶化）
- 完全不在论文中出现
- 复现时**完全跳过 v8 相关方法**，节省约 20% GPU 时间

### 5.7 torch-uncertainty 依赖已找到本地副本

- `10_research_archive/_archive/torch-uncertainty/src` 是 PackedEnsemble 依赖
- `_paths.py` 自动解析三个候选路径，目前指向上述位置
- 远程容器需要：要么 pip install torch-uncertainty，要么把 _archive 副本一起同步过去

### 5.8 Pre-existing fallbacks in UMC/ 已善用

- pretrain.py / train_sta_*.py 原本已有 `if os.path.isdir("/data/baiyimeng/dataset"): ... else: project_root/dataset` fallback 逻辑
- 简化为直接用 `_paths.DATA_ROOT`（env var + 30_reproduction/data/processed 兜底）
- train_neu_*.py 的 Config 类**也有同样 fallback**（之前漏改，第二轮 Edit 补上）

### 5.9 论文最终采纳的版本明确

- 论文 v1.13 数据来自 **v7+v7_supp+v9+v10 混合**（不依赖 v5/v6/v8）
- 详细映射见 plan §A.2 表格 + REVISION_LOG.md §V1.1

### 5.10 reproduction/utils/logging.py 与 stdlib 命名冲突（已规避）

**问题**：原本命名为 `reproduction/utils/logging.py`，导致 `python3 reproduction/utils/gpu.py` 直接运行时 sys.path[0] 指向 utils/ 目录，`import torch` 内部 `import logging` 被劫持到本地，AttributeError: module 'logging' has no attribute 'getLogger'。

**解决**：重命名为 `jsonl_log.py`，包级 export 名 `JsonlLogger` 保持不变。后续在 `reproduction/analysis/` 或 `reproduction/data/` 内新增子模块时**避免**使用 stdlib 重名：`logging`/`json`/`io`/`time`/`copy`/`types`/`platform` 等。

### 5.11 reproduction.utils 包级 export 列表（已稳定）

```python
from reproduction.utils import (
    setup_seed, derive_seed,           # seed.py
    detect_gpu, setup_hardware,        # gpu.py（含 dataloader_kwargs 未 export，可按需）
    JsonlLogger, jsonl_iter,           # jsonl_log.py
    write_status, update_phase,
    gpu_snapshot, disk_snapshot,
    DEFAULT_STATUS_PATH,               # status.py
)
```

后续 orchestrator / analysis 模块直接从顶层包导入。

### 5.12 UMC → reproduction.utils 反向 import 路径（关键）

UMC/train_*.py 在 main 开头将要 `from reproduction.utils.gpu import setup_hardware`（plan §F.1）。
当前 UMC/_paths.py L33 已暴露 `_PROJECT_ROOT = _HERE.parent`，后续 task #12 或 #16 修改 train_*.py 时需要把 `_PROJECT_ROOT` 加进 `sys.path`，否则 UMC 进程内无法找到 reproduction 包。已经手工验证过该 import 路径可行（无 cython 等编译产物，纯 Python）。

---

## 6. 验证当前已完成工作的命令

```bash
cd /Users/y/Research_MEM/30_reproduction

# 验证骨架完整性
find . -maxdepth 2 -type d | sort                     # 应有 24+ 子目录
ls -la *.md *.toml *.yml LICENSE .gitignore           # 5 个顶层文件
ls .githooks/pre-commit                                # hook 已就位（可执行）

# 验证 UMC 副本完整性
du -sh UMC                                             # ~1.9MB
ls UMC/_legacy_runs/                                   # 9 个 legacy 脚本 + archive/ + README.md
ls UMC/calib/ UMC/models/ UMC/utils/                   # 核心代码完整

# 验证 reproduction/utils/ (#15)
ls reproduction/utils/                                 # __init__.py + seed/gpu/jsonl_log/status.py
python3 reproduction/utils/seed.py                     # derive_seed 自检 OK
python3 reproduction/utils/gpu.py | head -8            # detect_gpu + setup_hardware 输出 OK
python3 reproduction/utils/jsonl_log.py                # JsonlLogger 写读自检 OK
python3 reproduction/utils/status.py                   # write_status + update_phase 自检 OK
python3 -c "from reproduction.utils import setup_seed, setup_hardware, JsonlLogger, write_status; print('package import OK')"

# 验证参数化（关键）
grep -rn "shared-nvme\|/data/baiyimeng" UMC/*.py       # 应为空（除 _paths.py docstring）
python3 UMC/_paths.py                                  # 应输出 3 个路径解析结果
for f in UMC/pretrain.py UMC/train_neu_*.py UMC/train_sta_*.py UMC/_paths.py; do
    python3 -c "import ast; ast.parse(open('$f').read())" && echo "OK $f"
done                                                   # 8 个文件应全 OK
```

预期所有命令通过。如有失败，**优先怀疑 Edit 操作未应用完全**，再排查其他。

---

## 7. 关键文件路径速查

### 7.1 项目核心文件

| 路径 | 用途 |
|------|------|
| `/Users/y/Research_MEM/30_reproduction/` | 本项目根 |
| `30_reproduction/README.md` | 项目首屏（一键复现命令 + 与论文 mapping）|
| `30_reproduction/CLAUDE.md` | 项目长期指令（本会话新建）|
| `30_reproduction/_SESSION_HANDOFF.md` | 本文件 |
| `30_reproduction/.gitignore` | 排除 data/ experiments/ logs/ |
| `30_reproduction/UMC/_paths.py` | **新增**：路径与依赖单点解析 |
| `30_reproduction/UMC/README_UMC.md` | UMC 目录归属与差异说明 |
| `30_reproduction/UMC/_legacy_runs/README.md` | 历史脚本说明 + 论文采纳关系 |

### 7.2 规划文件

| 路径 | 用途 |
|------|------|
| `/Users/y/.claude/plans/users-y-research-mem-00-active-artifact-validated-sedgewick.md` | **完整 plan（800+ 行）**，包含 §A 实验脉络、§A.4 复现哲学、§B 目录结构、§C 配置详表、§D RTX 5090、§G 本地-SSH、§M 验证方案 |

### 7.3 参考源（10_research_archive）

| 路径 | 用途 |
|------|------|
| `10_research_archive/UMC/` | 算法代码源（已 cp 到 30_reproduction/UMC/）|
| `10_research_archive/UMC/calib/MonotonicNN.py` | UMC + UAMCM + UASAC 核心算法 |
| `10_research_archive/ckpt/criteo/summary_all_meanstd.csv` | Ch4 主结果原数据 |
| `10_research_archive/ckpt/v9_error_analysis/samples/*.npz` | Ch3 sample-level NPZ（17 个，2GB）|
| `10_research_archive/ckpt/v10_ablation2/summary/*.csv` | Ch5 shuffled-u 消融 |
| `10_research_archive/_archive/torch-uncertainty/src` | PackedEnsemble 依赖（_paths.py 自动找到）|

### 7.4 论文权威源（00_active）

| 路径 | 用途 |
|------|------|
| `00_active/artifacts/v1.13_20260513/thesis.pdf` | 当前论文 PDF |
| `00_active/thesis/*.md` | 论文 markdown 源（回写目标）|
| `00_active/thesis/REVISION_LOG.md` | 数据验证日志（V1.1 含完整交叉验证表）|
| `00_active/CLAUDE.md` | 论文项目指令 |

---

## 8. 已知陷阱（避坑清单，复现时必查）

完整版见 plan §B，以下是最易踩的 3 条：

1. **batch_size 必须严守配置详表**：
   - pretrain: AliCCP/Criteo 32K, Avazu 16K
   - calib: AliCCP/Criteo 64K, Avazu 16K（**Avazu UAMCM 64K 会 OOM**）
   - 改了会导致数值漂移 + 与论文不可比

2. **ddof 严守 ddof=1**（Bessel 校正）：
   - 原 CSV 用 ddof=0 写入
   - 论文表 4-1 报告 ddof=1
   - 复现脚本统一用 ddof=1，与论文对齐

3. **ECE bins M=100**（不是常用 M=10）：
   - 论文用 M=100 捕捉细粒度校准误差
   - 改了与论文不可比

其他 7 条：
- 3 seed 制度 {1024, 2024, 3024} 不能改
- Loss-best 模型选择（不是 ECE-best）
- v8 DA-SCL 完全跳过（不收敛）
- Criteo seed 3024 极端值保留并标注
- field_index：AliCCP=0, Avazu=2, Criteo=23
- "ECE 降低 X%" 表述（禁用"改善 -X%"）
- cudnn.benchmark=False 必须保持

---

## 9. 本地-SSH 容器云协作模式（plan §G 摘要）

**核心约束**：远程 SSH 容器**无 Claude**，所有调试在本地。

**四层同步**：
- 代码：本地 git push → 远程 git pull
- 结果：远程 git push → 本地 git pull（仅 results/ 小文件）
- 元数据：rsync experiments/runs/**/{metrics.jsonl, done.flag, config.yaml}
- 日志：rsync logs/（节选 + 失败时手动）
- 状态包：远程每 5min 写 /root/status.json，本地 `ssh container cat /root/status.json | jq .`

**故障剧本**（plan §G.4 详述）：
1. 训练卡死 → ssh ps + tail log + nvidia-smi → kill + --resume
2. OOM → 本地改 configs/methods/*.yaml → push → 远程 pull → --resume
3. 容器回收 → 重申请 + git clone + setup_env + --resume（shared-nvme 保留 done.flag）
4. 数据下载失败 → aria2c retry + USTC/Kaggle 双源
等

**永远不入 git**：`*.pth` `*.npz` `data/raw/` `data/processed/` `experiments/runs/` `experiments/backbones/`

---

## 10. /clear vs /compact 建议

### 强烈推荐 /clear（理由）

| 维度 | /clear | /compact |
|------|--------|----------|
| 上下文状态 | 完全清空 | 智能压缩保留 |
| 信息无损保证 | ✓ 依赖本 HANDOFF + plan + CLAUDE.md（已 100% 落地）| ✗ 压缩可能丢失细节（如 P1-P5 论断的精确定义、避坑清单的根因）|
| 任务连续性 | TaskList 需用本 HANDOFF 重建 | TaskList 保留 |
| 新任务编码质量 | 清晰上下文，编码精度高 | 压缩后可能携带误差 |
| 适用场景 | **任务型推进**（剩 13 个独立任务）| 同一任务多轮迭代 |

### 推荐流程

```
1. 用 /clear 清空上下文
2. cd /Users/y/Research_MEM/30_reproduction/    # 自动加载本目录 CLAUDE.md
3. 给 Claude 第 0 节的开场提示词
4. Claude 阅读 HANDOFF + plan + 重建 TaskList 后开始任务 #15
```

### 不推荐 /compact 的额外理由

- 本会话 context 中含大量探索过程（多轮决策修正、Explore agent 输出等），压缩会保留这些**已过时的中间状态**
- /clear 让下次会话只看到"最终的、稳定的"信息：plan 文件 + HANDOFF + CLAUDE.md
- 13 个剩余任务都有清晰的输入输出契约，不需要"延续对话感"

### 何时考虑 /compact

仅当**单个任务跨多轮且未完成**时才用 /compact。当前所有完成任务都已落地到代码 + 文档，无未完成任务。

---

## 11. 自检清单（HANDOFF 完整性确认）

下次会话开始时，验证以下都能找到：

- [x] 项目目标（§1）
- [x] 所有锁定决策（§2）
- [x] 复现哲学（§3，关键）
- [x] 当前进度 3/16 任务 + 待办列表（§4）
- [x] 本次会话的非显然发现（§5）
- [x] 验证当前工作的命令（§6）
- [x] 关键文件路径速查（§7）
- [x] 避坑清单核心 10 条（§8 + plan §B）
- [x] 本地-SSH 协作模式（§9 + plan §G）
- [x] /clear vs /compact 建议（§10）
- [x] 开场提示词（§0）

如发现遗漏，更新本文件后再切换会话。

---

## 12. 维护约定

- 每完成一组任务（如完成 #15 后），更新本文件 §4 状态 + §5 新发现
- 每次会话切换前回到此文件确认更新
- 本文件**不入 git**（添加到 .gitignore？——目前没加，因为它是项目状态文档，可入 git）

**当前状态**：仅本地，未 git commit（因 30_reproduction 整个目录还未 git init，task #9 待做）。
