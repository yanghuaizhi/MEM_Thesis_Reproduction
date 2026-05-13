#!/usr/bin/env bash
# scripts/sync_to_local.sh — 远程容器 rsync 元数据 + 结果回本地
# 在本地 (macOS) 执行，远程容器只读访问。
set -euo pipefail

REMOTE_HOST="${1:-container}"        # 默认 ssh host alias
REMOTE_ROOT="${2:-/root/shared-nvme/30_reproduction}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== [sync] remote=$REMOTE_HOST:$REMOTE_ROOT  local=$LOCAL_ROOT ==="

# 1. 状态包（小，频繁）
ssh "$REMOTE_HOST" "cat /root/status.json" 2>/dev/null > "$LOCAL_ROOT/status.json" || \
    echo "[sync] WARN: status.json not available"

# 2. results/ (git tracked，应通过 git pull 拉)
echo "[sync] results/: use 'git pull' on local (远程已 git push)"

# 3. experiments 元数据（无 .pth, .npz）
rsync -av --include='*/' \
      --include='*.jsonl' \
      --include='*.yaml' \
      --include='*.json' \
      --include='done.flag' \
      --include='error.flag' \
      --exclude='*' \
      "$REMOTE_HOST:$REMOTE_ROOT/experiments/" \
      "$LOCAL_ROOT/experiments-meta/"

# 4. logs（节选 + 限大小）
rsync -av --max-size=5M \
      --include='*.log' \
      "$REMOTE_HOST:$REMOTE_ROOT/logs/" \
      "$LOCAL_ROOT/logs-mirror/" 2>/dev/null || true

echo "=== [sync] DONE ==="
echo "  - status.json: $LOCAL_ROOT/status.json"
echo "  - meta:        $LOCAL_ROOT/experiments-meta/"
echo "  - logs:        $LOCAL_ROOT/logs-mirror/"
