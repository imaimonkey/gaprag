from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .utils import ensure_dir, save_json, timestamp


def create_run_dir(base_dir: str | Path = "outputs/runs", run_name: str | None = None) -> Path:
    base = ensure_dir(base_dir)
    name = run_name or timestamp()
    run_dir = ensure_dir(base / name)
    return run_dir


def setup_logger(log_path: str | Path | None = None, name: str = "gapverify") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        path = Path(log_path)
        ensure_dir(path.parent)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def snapshot_config(config: dict[str, Any], run_dir: str | Path) -> None:
    save_json(config, Path(run_dir) / "config_snapshot.json")
