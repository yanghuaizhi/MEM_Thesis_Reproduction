# RUNBOOK — 10 个故障应对剧本

> 远程 SSH 容器**无 Claude**。任何故障按本文档执行。
> 来源: plan §G.4 + 实际复现经验。

---

## 剧本 1: 训练卡死

**症状**: status.json 的 epoch/iter 30 分钟无变化。

**诊断**:
```bash
ssh container 'ps aux | grep python | head'              # 找进程
ssh container 'tail -100 /root/shared-nvme/30_reproduction/logs/current.log'
ssh container 'nvidia-smi'                                # GPU 状态
```

**操作**:
```bash
# 确认卡死后
ssh container 'kill <pid>'
ssh container 'cd /root/shared-nvme/30_reproduction && bash scripts/run_main_experiments.sh --resume'
```

---

## 剧本 2: OOM (CUDA out of memory)

**症状**: log 中 `CUDA out of memory`，常见于 Avazu + UAMCM/UASAC + 64K batch。

**根因**: plan §B 第 5 条避坑——Avazu calib 必须 16K。

**操作**:
```bash
# 本地检查配置
cat reproduction/configs/datasets/avazu.yaml | grep calib  # 应该 16384

# 如果配置错了，本地改 + push + 远程 pull + resume
git commit -am "fix: lower avazu calib batch to 16K (OOM)"
git push
ssh container 'cd /root/shared-nvme/30_reproduction && git pull && bash scripts/run_main_experiments.sh --resume --dataset avazu'
```

---

## 剧本 3: 容器被回收

**症状**: ssh 连接超时，平台显示容器已停止。

**操作**:
1. 平台重新申请同规格容器（RTX 5090 32GB 山东二区）
2. ssh 进入新容器
3. `cd /root/shared-nvme/ && ls`  # 共享存储应保留所有 done.flag
4. 如果 30_reproduction/ 还在: `cd 30_reproduction && bash scripts/setup_env.sh`
   否则: `git clone https://github.com/yanghuaizhi/MEM_Thesis_Reproduction.git 30_reproduction`
5. `bash scripts/run_main_experiments.sh --resume`（跳过 done.flag）

---

## 剧本 4: 数据下载失败

**症状**: `bash scripts/download_data.sh` 报某文件 md5 失败或下载 0 字节。

**操作**:
```bash
# 1. 看哪些文件 missing
ssh container 'python3 -m reproduction.data.download --verify-only'

# 2. 重新跑 USTC / Kaggle 下载（按 download.py 输出的指南）
# USTC 链接需要浏览器登录密码 5277
# Kaggle 需要 ~/.kaggle/kaggle.json
```

如果 USTC 链接挂了，切到 Kaggle 备源（详见 data/README.md §下载源）。

---

## 剧本 5: 依赖冲突

**症状**: pip install 报版本冲突，或 import torch_uncertainty 失败。

**操作**:
```bash
# 1. 重装关键包
ssh container 'pip install --upgrade torch>=2.7.0 numpy pandas scikit-learn pyyaml'

# 2. torch-uncertainty 单独处理（如果 pip 装失败）
ssh container 'pip install torch-uncertainty'
# 或者用本地 _archive 副本:
rsync -av 10_research_archive/_archive/torch-uncertainty/ container:/root/torch-uncertainty/
ssh container 'export MEM_TORCH_UNCERTAINTY_SRC=/root/torch-uncertainty/src'
```

---

## 剧本 6: git push 冲突

**症状**: 本地 push 失败 "rejected, non-fast-forward"。

**根因**: 远程容器先 push 了 results/。

**操作**:
```bash
git pull --rebase                          # 本地接受远程 results/
# 解决任何代码冲突
git push
```

---

## 剧本 7: 磁盘满

**症状**: `No space left on device`。

**诊断**:
```bash
ssh container 'df -h /root/shared-nvme'
ssh container 'du -sh /root/shared-nvme/30_reproduction/*'
```

**操作**:
```bash
# 删旧 backbone（已用过的 seed=1024 backbone，可重生）
ssh container 'find /root/shared-nvme/30_reproduction/experiments/runs/pretrain -name "*.pth" -mtime +7 -delete'

# 删失败的 run 残留
ssh container 'find /root/shared-nvme/30_reproduction/experiments/runs -name "error.flag" -exec rm -rf {}/.. \;'

# 清 pip cache
ssh container 'pip cache purge'
```

---

## 剧本 8: 网络断（rsync/ssh 卡）

**症状**: ssh 命令 hang 或 rsync 长时间不动。

**操作**:
```bash
# 设 SSH 超时
ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 container 'echo alive'

# 如果远程任务还在跑（status.json 还在更新），等网络恢复即可
# 如果挂了，按剧本 1 处理
```

---

## 剧本 9: Checkpoint 损坏

**症状**: 加载 `.pth` 时报 `RuntimeError: invalid magic number` 或形状不匹配。

**操作**:
```bash
# 1. 删坏 checkpoint + done.flag
ssh container 'rm /root/shared-nvme/30_reproduction/experiments/runs/pretrain/<dataset>/_backbone/seed_<N>/{*.pth,done.flag}'

# 2. 触发重跑
ssh container 'bash scripts/run_pretrain.sh --resume --dataset <X> --seed <N>'
```

如果是 calib checkpoint 坏: 同样删 `done.flag` 后 `--resume` 即可。

---

## 剧本 10: 结果异常（sanity_check 失败）

**症状**: `sanity_check --strict` 退出码 1，报告某 run 的 ECE 超范围或 CV 爆炸。

**操作**:
```bash
# 1. 看具体哪些 run 有问题
ssh container 'cat /root/shared-nvme/30_reproduction/results/diff_audit/sanity_check_report.md'

# 2. 看那个 run 的日志
ssh container 'cat /root/shared-nvme/30_reproduction/experiments/runs/main/<ds>/<method>/seed_<N>/train.log'

# 3. 根因判断:
#    - field_index 错 → 配置 bug (改 configs/datasets/*.yaml)
#    - ECE 超范围 → 训练发散 (lr/init 问题)
#    - CV 爆炸 → 单 seed 极端值 (plan §B 第 4 条 Criteo seed 3024 已知)

# 4. 修复后:
git push
ssh container 'git pull'
ssh container 'rm -rf .../<问题 run dir>'   # 不可逆，请确认
ssh container 'bash scripts/run_main_experiments.sh --resume'
```

---

## 紧急联系与参考

| 资源 | 用途 |
|------|------|
| `docs/07_known_issues.md` | 已知坑清单 |
| `docs/09_local_ssh_workflow.md` | 同步机制详解 |
| `results/diff_audit/sanity_check_report.md` | M.0 质量门状态 |
| plan §G.4 | 完整故障树 |

**禁止动作（plan production-safety）**:
- 远程容器**不要**手动 INSERT/UPDATE 任何文件，除非剧本明确允许
- 不要 `--no-verify` 跳过 git hook
- 不要 `git push --force` 主分支
