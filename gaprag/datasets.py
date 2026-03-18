from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _load_local_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict) and "data" in loaded and isinstance(loaded["data"], list):
            return loaded["data"]
        raise ValueError(f"Unsupported JSON shape: {path}")
    raise ValueError(f"Unsupported file type: {path}")


def load_corpus(path: str) -> list[dict[str, Any]]:
    records = _load_local_json_or_jsonl(Path(path))
    normalized: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        doc_id = str(rec.get("id", i))
        text = str(rec.get("text", rec.get("content", ""))).strip()
        if not text:
            continue
        normalized.append({"id": doc_id, "text": text, "metadata": rec.get("metadata", {})})
    return normalized


def _normalize_qa_item(item: dict[str, Any], idx: int, field_map: dict[str, str]) -> dict[str, Any]:
    q_key = field_map.get("question", "question")
    a_key = field_map.get("answers", "answers")
    id_key = field_map.get("id", "id")
    ctx_key = field_map.get("context", "context")
    md_key = field_map.get("metadata", "metadata")
    sess_key = field_map.get("session_id", "session_id")

    answers_raw = item.get(a_key, [])
    if isinstance(answers_raw, str):
        answers = [answers_raw]
    elif isinstance(answers_raw, list):
        answers = [str(x) for x in answers_raw]
    else:
        answers = [str(answers_raw)] if answers_raw is not None else []

    context = item.get(ctx_key, [])
    if context is None:
        context = []
    if not isinstance(context, list):
        context = [context]

    metadata = item.get(md_key, {})
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}

    session_id = str(item.get(sess_key, metadata.get("session_id", "default")))

    return {
        "id": str(item.get(id_key, idx)),
        "question": str(item.get(q_key, "")),
        "answers": answers,
        "context": [str(x) for x in context],
        "metadata": metadata,
        "session_id": session_id,
    }


def load_qa_dataset(config: dict[str, Any]) -> list[dict[str, Any]]:
    ds_cfg = config.get("dataset", {})
    field_map = ds_cfg.get("field_map", {})

    local_path = ds_cfg.get("path")
    if local_path:
        rows = _load_local_json_or_jsonl(Path(local_path))
        normalized = [_normalize_qa_item(r, i, field_map) for i, r in enumerate(rows)]
    else:
        name = ds_cfg.get("hf_name")
        if not name:
            raise ValueError("dataset.path or dataset.hf_name must be provided")
        split = ds_cfg.get("split", "validation")
        subset = ds_cfg.get("hf_subset")
        if subset:
            ds = load_dataset(name, subset, split=split)
        else:
            ds = load_dataset(name, split=split)
        normalized = [_normalize_qa_item(dict(row), i, field_map) for i, row in enumerate(ds)]

    limit = int(ds_cfg.get("limit", 0) or 0)
    if limit > 0:
        normalized = normalized[:limit]

    return normalized
