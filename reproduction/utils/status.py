"""状态包写入：远程容器每 5min 写 /root/status.json，本地 ssh + jq 读取。

Schema（plan §G.3）:
    {
      "timestamp": "2026-05-14T03:42:11+0800",
      "phase": "stage_4_uamcm",
      "current_run": {"dataset": "aliccp", "method": "uamcm", "seed": 2024,
                      "epoch": 12, "iter": 8400},
      "gpu": {"util": "94%", "mem_used_mb": 18432, "mem_total_mb": 32768,
              "temp_c": 71},
      "disk": {"shared_nvme_used_gb": 52, "shared_nvme_free_gb": 28},
      "budget": {"hours_used": 41.3, "hours_total": 114,
                 "hours_remaining": 72.7, "rmb_spent": 123},
      "last_error": null,
      "next_eta": {"current_run_done": "+18min", "stage_done": "+4.2h",
                   "all_done": "+38h"},
      "done_flags": 47
    }

写入策略:
    - atomic（写 .tmp 后 os.replace）
    - update_phase 做 read-modify-write 合并
    - macOS / 无 nvidia-smi 时 gpu_snapshot 返回空字典
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]

DEFAULT_STATUS_PATH = os.environ.get("MEM_STATUS_PATH", "/root/status.json")


def write_status(state: Dict[str, Any], path: Optional[PathLike] = None) -> Path:
    """原子写入 status snapshot。state 中如无 timestamp 则自动注入。

    Returns:
        实际写入的 Path（便于调用方做后续操作）。
    """
    target = Path(path or DEFAULT_STATUS_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"timestamp": _iso_now()}
    payload.update(state)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, target)
    return target


def read_status(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """读 status.json，文件不存在或损坏时返回 {}."""
    p = Path(path or DEFAULT_STATUS_PATH)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def update_phase(
    phase: str,
    path: Optional[PathLike] = None,
    **extra: Any,
) -> Path:
    """合并方式更新 phase + 其它字段。读旧 → 合并 → 写新。"""
    cur = read_status(path)
    cur["phase"] = phase
    cur.update(extra)
    return write_status(cur, path)


def gpu_snapshot() -> Dict[str, Any]:
    """通过 nvidia-smi 抓 GPU 实时状态。无该命令则返回空字典。"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        if not out:
            return {}
        parts = [x.strip() for x in out.splitlines()[0].split(",")]
        if len(parts) < 4:
            return {}
        util, mem_used, mem_total, temp = parts[:4]
        return {
            "util": f"{int(util)}%",
            "mem_used_mb": int(mem_used),
            "mem_total_mb": int(mem_total),
            "temp_c": int(temp),
        }
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    except Exception:
        return {}


def disk_snapshot(path: PathLike = "/root/shared-nvme") -> Dict[str, Any]:
    """返回 path 所在磁盘的 used/free GB。路径不存在则空字典。"""
    import shutil

    p = Path(path)
    if not p.exists():
        return {}
    try:
        usage = shutil.disk_usage(str(p))
        key_used = f"{p.name or 'root'}_used_gb"
        key_free = f"{p.name or 'root'}_free_gb"
        return {
            key_used: round((usage.total - usage.free) / 1024**3, 1),
            key_free: round(usage.free / 1024**3, 1),
        }
    except OSError:
        return {}


def _iso_now() -> str:
    """ISO8601 + 北京时区（CST，UTC+8），便于跨区域容器读取一致。"""
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # 1) write_status
        write_status(
            {
                "phase": "stage_4_uamcm",
                "current_run": {"dataset": "aliccp", "method": "uamcm", "seed": 2024},
                "gpu": gpu_snapshot(),
                "disk": disk_snapshot(Path(tmp_path).parent),
            },
            path=tmp_path,
        )
        s = read_status(tmp_path)
        assert s["phase"] == "stage_4_uamcm"
        assert "timestamp" in s
        # 2) update_phase
        update_phase("stage_5_v9", path=tmp_path, done_flags=42)
        s = read_status(tmp_path)
        assert s["phase"] == "stage_5_v9"
        assert s["done_flags"] == 42
        # current_run 应保留（update_phase 不清空）
        assert s["current_run"]["method"] == "uamcm"
        print(f"status.py self-check OK; sample snapshot:")
        print(json.dumps(s, indent=2, ensure_ascii=False))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
