"""结构化日志：jsonl 写入 + stdout 镜像。

每条记录是一行 JSON（含 ts + event + 任意 fields），便于：
    - tail -f 时人看
    - jq/pandas 读 metrics.jsonl 聚合
    - rsync 拉回本地后离线分析

典型用法:
    with JsonlLogger("experiments/runs/aliccp/umc/seed_1024/metrics.jsonl") as L:
        L.log("epoch_end", epoch=3, ece=0.0732, logloss=0.4521)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Union


PathLike = Union[str, Path]


class JsonlLogger:
    """append-only jsonl writer，每条记录单独 flush 以耐受 kill。"""

    def __init__(
        self,
        path: PathLike,
        also_stdout: bool = True,
        flush_every: int = 1,
        stdout_prefix: str = "",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.also_stdout = also_stdout
        self.flush_every = max(1, int(flush_every))
        self.stdout_prefix = stdout_prefix
        self._count = 0
        self._fp = open(self.path, "a", buffering=1)  # line buffered

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        self._fp.write(line + "\n")
        self._count += 1
        if self._count % self.flush_every == 0:
            self._fp.flush()
        if self.also_stdout:
            kv = " ".join(f"{k}={_short(v)}" for k, v in fields.items())
            print(
                f"{self.stdout_prefix}[{event}] {kv}".rstrip(),
                file=sys.stdout,
                flush=True,
            )

    def close(self) -> None:
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # 异常时记录一笔便于排查
        if exc_type is not None:
            try:
                self.log("logger_exit_error", error_type=exc_type.__name__, error=str(exc))
            except Exception:
                pass
        self.close()


def jsonl_iter(path: PathLike) -> Iterator[dict]:
    """逐行读 jsonl，容忍空行 / 损坏行（跳过并打印 stderr）。"""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as e:
                print(
                    f"[jsonl_iter] WARN: {p}:{lineno} JSON decode error: {e}",
                    file=sys.stderr,
                )


def _json_default(obj: Any) -> Any:
    """fallback 序列化：numpy / pathlib / set 等。"""
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def _short(v: Any, maxlen: int = 80) -> str:
    s = repr(v) if not isinstance(v, (int, float, str, bool)) else str(v)
    if isinstance(v, float):
        s = f"{v:.6g}"
    if len(s) > maxlen:
        s = s[: maxlen - 3] + "..."
    return s


if __name__ == "__main__":
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with JsonlLogger(tmp_path, stdout_prefix="[test] ") as L:
            L.log("epoch_start", epoch=0)
            L.log("epoch_end", epoch=0, ece=0.07321, logloss=0.4521)
            L.log("note", msg="non-numeric value")
        records = list(jsonl_iter(tmp_path))
        assert len(records) == 3, f"expected 3 records, got {len(records)}"
        assert records[1]["event"] == "epoch_end"
        assert abs(records[1]["ece"] - 0.07321) < 1e-9
        print(f"\njsonl_log.py self-check OK ({len(records)} records)")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
