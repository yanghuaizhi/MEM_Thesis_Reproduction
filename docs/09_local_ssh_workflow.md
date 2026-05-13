# 09: 本地-SSH 协作模式（plan §G）

> 远程 SSH 容器**无 Claude**。本地 (macOS, Claude 可用) 负责诊断/修复/分析，
> 远程容器只跑训练。需要"本地脑 + 远程手"的协作模式。

## 1. 四层同步架构

```
本地 (macOS, Claude 可用)              远程 SSH 容器 (RTX 5090, 纯执行)
─────────────────────────              ──────────────────────────────
30_reproduction/                       /root/shared-nvme/30_reproduction/
  ├── UMC/                             ├── UMC/
  ├── reproduction/        ─git push─▶ ├── reproduction/
  ├── scripts/             ─git pull─▶ ├── scripts/
  ├── docs/                ──────────▶ ├── docs/
  │                                    │
  ├── results/             ◀─git push─ ├── results/        (远程聚合后入 git)
  ├── experiments-meta/    ◀──rsync─── ├── experiments/    (metrics.jsonl 等小文件)
  ├── logs-mirror/         ◀──rsync─── │   └── logs/
  │                                    │
  ├── status.json          ◀──ssh cat─ └── /root/status.json   (状态包)
```

## 2. 五种同步通道

| 通道 | 方向 | 内容 | 触发时机 | 工具 |
|------|------|------|---------|------|
| 代码 | 本地→远程 | UMC/ + reproduction/ + scripts/ + configs/ | 本地 commit 后 | git push/pull |
| 结果（小）| 远程→本地 | results/{tables,figures}/*.md, *.csv, *.pdf | 远程聚合后 | git push/pull |
| 元数据（中）| 远程→本地 | metrics.jsonl + done.flag + config.yaml | 阶段完成时 | rsync |
| 日志（中）| 远程→本地 | logs/*.log 节选 | 30min 自动 + 失败时手动 | rsync |
| 大产物 | 远程→本地 | predictions.npz, samples.npz | 分析时手动 | rsync 单文件 |

**永远不入 git**: `*.pth`、`data/raw/`、`data/processed/`、`experiments/runs/`。

## 3. 远程状态包（plan §G.3）

远程容器后台跑 `bash scripts/health_check.sh &`，每 5 min 写 `/root/status.json`:

```json
{
  "timestamp": "2026-05-14T03:42:11+0800",
  "phase": "stage_4_uamcm",
  "current_run": {"dataset": "aliccp", "method": "uamcm", "seed": 2024, "epoch": 12},
  "gpu": {"util": "94%", "mem_used_mb": 18432, "mem_total_mb": 32768, "temp_c": 71},
  "disk": {"shared_nvme_used_gb": 52, "shared_nvme_free_gb": 28},
  "budget": {"hours_used": 41.3, "hours_total": 114, "rmb_spent": 123},
  "done_flags": 47
}
```

**本地一行查看**:
```bash
ssh container 'cat /root/status.json' | jq .
```

## 4. 本地监控命令

```bash
# 实时进度
watch -n 60 'ssh container "cat /root/status.json" | jq ".phase, .current_run, .budget"'

# 实时日志
ssh container 'tail -f /root/shared-nvme/30_reproduction/logs/current.log'

# 拉小文件（仅 metadata，不拉 .pth）
bash scripts/sync_to_local.sh container

# 跑完后拉 results
git pull
```

## 5. 修复 → 同步循环

发现远程问题:
1. 本地 Claude 诊断（看 status.json + log）
2. 本地修复代码 / 配置
3. `git commit -m "fix: X"` + `git push`
4. 远程 `git pull`
5. 远程 `bash scripts/run_*.sh --resume`（done.flag 跳过已完成 run）

## 6. 故障应对

详见 [RUNBOOK.md](RUNBOOK.md) — 10 个剧本覆盖训练卡死/OOM/容器回收/数据下载
失败/依赖冲突/git push 冲突/磁盘满/网络断/checkpoint 损坏/结果异常。

## 7. SSH alias 配置（本地 ~/.ssh/config）

```
Host container
    HostName <remote_ip>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
    StrictHostKeyChecking accept-new
```

## 8. 安全注意

- 本地容量有限：data/ 不下载本地，仅查元信息（feature_meta.json、md5）
- 大文件分析时远程随机抽 1000 行写 CSV → rsync 拉回
- `*.pth` backbone 文件**绝不**进入本地（每个 7GB+）
