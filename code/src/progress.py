from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "-"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressTracker:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path = self.path.with_name(f"{self.path.stem}_history.jsonl")
        self.svg_path = self.path.with_name(f"{self.path.stem}.svg")
        self.state: dict[str, Any] = {}

    def start(self, stage: str, total: int, extra: dict[str, Any] | None = None) -> None:
        started_at = now_iso()
        self.state = {
            "stage": stage,
            "status": "running",
            "total": int(total),
            "processed": 0,
            "success": 0,
            "failed": 0,
            "percent": 0.0,
            "current_item": "",
            "last_error": "",
            "started_at": started_at,
            "updated_at": started_at,
            "elapsed_seconds": 0.0,
            "items_per_second": 0.0,
            "eta_seconds": None,
            "eta_hms": "-",
            "elapsed_hms": "00:00",
            "history_file": str(self.history_path),
            "progress_svg": str(self.svg_path),
        }
        if extra:
            self.state.update(extra)
        self.history_path.write_text("", encoding="utf-8")
        self._write()
        self._append_history()
        self._write_svg()

    def update(
        self,
        processed: int,
        success: int,
        failed: int,
        current_item: str,
        last_error: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        updated_at = now_iso()
        self.state.update(
            {
                "processed": int(processed),
                "success": int(success),
                "failed": int(failed),
                "current_item": current_item,
                "last_error": last_error,
                "updated_at": updated_at,
            }
        )
        self._refresh_timing()
        if extra:
            self.state.update(extra)
        self._write()
        self._append_history()
        self._write_svg()

    def finish(self, status: str = "completed", extra: dict[str, Any] | None = None) -> None:
        finished_at = now_iso()
        self.state["status"] = status
        self.state["updated_at"] = finished_at
        self.state["finished_at"] = finished_at
        self._refresh_timing()
        self.state["eta_seconds"] = 0.0
        self.state["eta_hms"] = "00:00"
        if extra:
            self.state.update(extra)
        self._write()
        self._append_history()
        self._write_svg()

    def tqdm_postfix(self) -> dict[str, str]:
        return {
            "ok": str(self.state.get("success", 0)),
            "fail": str(self.state.get("failed", 0)),
            "rate": f"{float(self.state.get('items_per_second', 0.0)):.2f}/s",
            "eta": str(self.state.get("eta_hms", "-")),
        }

    def _refresh_timing(self) -> None:
        started_at = parse_iso(str(self.state.get("started_at", "")))
        updated_at = parse_iso(str(self.state.get("updated_at", "")))
        total = max(int(self.state.get("total", 0)), 1)
        processed = int(self.state.get("processed", 0))
        if not started_at or not updated_at:
            return
        elapsed_seconds = max((updated_at - started_at).total_seconds(), 0.0)
        items_per_second = processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        remaining = max(total - processed, 0)
        eta_seconds = (remaining / items_per_second) if items_per_second > 0 else None
        self.state["percent"] = round(processed * 100.0 / total, 2)
        self.state["elapsed_seconds"] = round(elapsed_seconds, 2)
        self.state["items_per_second"] = round(items_per_second, 4)
        self.state["eta_seconds"] = round(eta_seconds, 2) if eta_seconds is not None else None
        self.state["elapsed_hms"] = format_duration(elapsed_seconds)
        self.state["eta_hms"] = format_duration(eta_seconds)

    def _append_history(self) -> None:
        snapshot = {
            "timestamp": self.state.get("updated_at", now_iso()),
            "stage": self.state.get("stage", ""),
            "status": self.state.get("status", ""),
            "processed": self.state.get("processed", 0),
            "total": self.state.get("total", 0),
            "success": self.state.get("success", 0),
            "failed": self.state.get("failed", 0),
            "percent": self.state.get("percent", 0.0),
            "items_per_second": self.state.get("items_per_second", 0.0),
            "eta_seconds": self.state.get("eta_seconds", None),
            "current_item": self.state.get("current_item", ""),
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    def _write_svg(self) -> None:
        lines = []
        if self.history_path.exists():
            for raw_line in self.history_path.read_text(encoding="utf-8").splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    lines.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    continue
        if not lines:
            return

        width = 900
        height = 320
        left = 70
        right = 30
        top = 30
        bottom = 70
        plot_width = width - left - right
        plot_height = height - top - bottom

        xs = []
        ys = []
        total_points = max(len(lines) - 1, 1)
        for index, item in enumerate(lines):
            percent = float(item.get("percent", 0.0))
            x = left + plot_width * index / total_points
            y = top + plot_height * (1.0 - min(max(percent, 0.0), 100.0) / 100.0)
            xs.append(x)
            ys.append(y)
        path_data = " ".join(
            ("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
            for index, (x, y) in enumerate(zip(xs, ys))
        )
        latest = lines[-1]
        rate = float(latest.get("items_per_second", 0.0))
        eta_hms = format_duration(latest.get("eta_seconds", None))
        percent = float(latest.get("percent", 0.0))
        stage = str(latest.get("stage", "progress"))

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7f8fb"/>
  <text x="{left}" y="20" font-family="Segoe UI, Microsoft YaHei UI, sans-serif" font-size="18" font-weight="600" fill="#1f2937">{stage}</text>
  <text x="{left}" y="{height - 38}" font-family="Consolas, monospace" font-size="14" fill="#374151">Progress {percent:.2f}%</text>
  <text x="{left + 210}" y="{height - 38}" font-family="Consolas, monospace" font-size="14" fill="#374151">Rate {rate:.2f}/s</text>
  <text x="{left + 390}" y="{height - 38}" font-family="Consolas, monospace" font-size="14" fill="#374151">ETA {eta_hms}</text>
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#cbd5e1" stroke-width="1"/>
  <line x1="{left}" y1="{top}" x2="{left + plot_width}" y2="{top}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4 4"/>
  <line x1="{left}" y1="{top + plot_height / 2}" x2="{left + plot_width}" y2="{top + plot_height / 2}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4 4"/>
  <text x="18" y="{top + 5}" font-family="Consolas, monospace" font-size="12" fill="#6b7280">100%</text>
  <text x="26" y="{top + plot_height / 2 + 5}" font-family="Consolas, monospace" font-size="12" fill="#6b7280">50%</text>
  <text x="34" y="{top + plot_height + 5}" font-family="Consolas, monospace" font-size="12" fill="#6b7280">0%</text>
  <path d="{path_data}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="{xs[-1]:.2f}" cy="{ys[-1]:.2f}" r="4" fill="#ef4444"/>
</svg>
"""
        self.svg_path.write_text(svg, encoding="utf-8")

    def _write(self) -> None:
        self.path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
