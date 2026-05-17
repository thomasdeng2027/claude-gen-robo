"""Minimal per-run logger for the keypoint pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return str(x)


class PipelineLogger:
    def __init__(self, base_dir: str = "runs/keypoint_pipeline"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"run_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.iterations: list[dict] = []
        print(f"[logger] run_dir = {self.run_dir}")

    def log_iteration(
        self,
        iteration: int,
        generated_code: str,
        stdout: str,
        stderr: str,
        success: bool,
        **extra: Any,
    ) -> Path:
        d = self.run_dir / f"iteration_{iteration:02d}"
        d.mkdir(parents=True, exist_ok=True)
        # generated_code.py / stdout.txt / stderr.txt are written by pipeline.py;
        # the logger only owns metadata.json.
        meta = {"iteration": iteration, "success": bool(success)}
        meta.update({k: _jsonable(v) for k, v in extra.items()})
        (d / "metadata.json").write_text(json.dumps(meta, indent=2))
        self.iterations.append(meta)
        return d

    def finalize(self, success: bool, total_iterations: int) -> None:
        summary = {
            "success": bool(success),
            "total_iterations": int(total_iterations),
            "iterations": self.iterations,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[logger] summary → {self.run_dir / 'summary.json'}")
