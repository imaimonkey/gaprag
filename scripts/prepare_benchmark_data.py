#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from datasets import load_dataset

from gaprag.utils import save_json, save_jsonl


def _slug(text: str, max_len: int = 48) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    if not value:
        value = "na"
    return value[:max_len]


def _doc_id(prefix: str, key: str) -> str:
    digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}::{_slug(key, max_len=40)}::{digest}"


def _dedup_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


@dataclass
class PreparedSplit:
    benchmark: str
    corpus: list[dict[str, Any]]
    qa: list[dict[str, Any]]


def _prepare_nq(limit: int, split: str) -> PreparedSplit:
    ds = load_dataset("cjlovering/natural-questions-short", split=split)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    docs: dict[str, dict[str, Any]] = {}
    qa_rows: list[dict[str, Any]] = []

    for i, item in enumerate(ds):
        questions = item.get("questions", [])
        question = ""
        if isinstance(questions, list) and questions:
            q0 = questions[0]
            if isinstance(q0, dict):
                question = str(q0.get("input_text", "")).strip()
            else:
                question = str(q0).strip()
        if not question:
            continue

        answers = item.get("answers", [])
        answer_texts: list[str] = []
        if isinstance(answers, list):
            for ans in answers:
                if isinstance(ans, dict):
                    span = str(ans.get("span_text", "")).strip()
                    if span:
                        answer_texts.append(span)
                elif ans is not None:
                    answer_texts.append(str(ans))
        answer_texts = _dedup_list(answer_texts)
        if not answer_texts:
            continue

        context = str(item.get("contexts", "")).strip()
        if not context:
            continue

        item_id = str(item.get("id", i))
        title = str(item.get("name", "nq"))
        did = _doc_id("nq", item_id)
        docs[did] = {
            "id": did,
            "text": context,
            "metadata": {
                "benchmark": "nq",
                "source_id": item_id,
                "title": title,
            },
        }
        qa_rows.append(
            {
                "id": f"nq::{item_id}",
                "question": question,
                "answers": answer_texts,
                "context": [did],
                "metadata": {
                    "benchmark": "nq",
                    "source_id": item_id,
                    "title": title,
                },
                "session_id": f"nq::{_slug(title)}",
            }
        )

    return PreparedSplit(benchmark="nq", corpus=list(docs.values()), qa=qa_rows)


def _prepare_hotpotqa(limit: int, split: str) -> PreparedSplit:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    docs: dict[str, dict[str, Any]] = {}
    qa_rows: list[dict[str, Any]] = []

    for i, item in enumerate(ds):
        qid = str(item.get("id", i))
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue

        ctx = item.get("context", {})
        titles = list(ctx.get("title", [])) if isinstance(ctx, dict) else []
        sentences = list(ctx.get("sentences", [])) if isinstance(ctx, dict) else []
        title_to_id: dict[str, str] = {}
        context_doc_ids: list[str] = []

        for title, sent_list in zip(titles, sentences):
            did = _doc_id("hotpot", str(title))
            text = " ".join(str(s).strip() for s in sent_list if str(s).strip())
            if not text:
                continue
            title_to_id[str(title)] = did
            context_doc_ids.append(did)
            if did not in docs:
                docs[did] = {
                    "id": did,
                    "text": text,
                    "metadata": {
                        "benchmark": "hotpotqa",
                        "title": str(title),
                    },
                }

        support = item.get("supporting_facts", {})
        support_titles = list(support.get("title", [])) if isinstance(support, dict) else []
        gold_ids = [title_to_id[t] for t in support_titles if t in title_to_id]
        gold_ids = _dedup_list(gold_ids)
        if not gold_ids:
            gold_ids = context_doc_ids[:2]
        if not gold_ids:
            continue

        session_key = support_titles[0] if support_titles else titles[0] if titles else "hotpot"
        qa_rows.append(
            {
                "id": f"hotpot::{qid}",
                "question": question,
                "answers": [answer],
                "context": gold_ids,
                "metadata": {
                    "benchmark": "hotpotqa",
                    "type": str(item.get("type", "")),
                    "level": str(item.get("level", "")),
                },
                "session_id": f"hotpot::{_slug(session_key)}",
            }
        )

    return PreparedSplit(benchmark="hotpotqa", corpus=list(docs.values()), qa=qa_rows)


def _prepare_fever(limit: int, split: str) -> PreparedSplit:
    ds = load_dataset("copenlu/fever_gold_evidence", split=split)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    docs: dict[str, dict[str, Any]] = {}
    qa_rows: list[dict[str, Any]] = []

    for i, item in enumerate(ds):
        source_id = str(item.get("id", i))
        claim = str(item.get("claim", "")).strip()
        label = str(item.get("label", "")).strip().upper()
        if not claim or not label:
            continue

        evidence = item.get("evidence", [])
        gold_ids: list[str] = []
        session_title = "fever"
        for ev in evidence:
            if not isinstance(ev, list) or len(ev) < 3:
                continue
            title = str(ev[0]).strip()
            sent_id = str(ev[1]).strip()
            sent_text = str(ev[2]).strip()
            if not title or not sent_text:
                continue
            session_title = title
            did = _doc_id("fever", f"{title}::{sent_id}")
            gold_ids.append(did)
            if did not in docs:
                docs[did] = {
                    "id": did,
                    "text": sent_text,
                    "metadata": {
                        "benchmark": "fever",
                        "title": title,
                        "sentence_id": sent_id,
                    },
                }

        gold_ids = _dedup_list(gold_ids)
        if not gold_ids:
            continue

        qa_rows.append(
            {
                "id": f"fever::{source_id}",
                "question": claim,
                "answers": [label],
                "context": gold_ids,
                "metadata": {
                    "benchmark": "fever",
                    "verifiable": str(item.get("verifiable", "")),
                },
                "session_id": f"fever::{_slug(session_title)}",
            }
        )

    return PreparedSplit(benchmark="fever", corpus=list(docs.values()), qa=qa_rows)


def _annotate_rows(name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for base in rows:
        row = dict(base)
        md = dict(base.get("metadata", {}))
        md["benchmark"] = name
        md["orig_session_id"] = str(base.get("session_id", "default"))
        row["metadata"] = md
        row["session_id"] = f"{name}::{_slug(base.get('session_id', 'default'))}"
        annotated.append(row)
    return annotated


def _build_continual_rows(
    named_rows: list[tuple[str, list[dict[str, Any]]]],
    min_session_size: int = 2,
    order: str = "session_grouped",
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for name, rows in named_rows:
        annotated.extend(_annotate_rows(name, rows))

    session_rows: dict[str, list[dict[str, Any]]] = {}
    for row in annotated:
        session_rows.setdefault(str(row["session_id"]), []).append(row)

    filtered = {
        sid: rows
        for sid, rows in session_rows.items()
        if len(rows) >= int(max(min_session_size, 1))
    }
    if not filtered:
        return []

    def _decorate(row: dict[str, Any], session_size: int, session_step: int) -> dict[str, Any]:
        out = dict(row)
        md = dict(row.get("metadata", {}))
        md["session_size"] = session_size
        md["session_step"] = session_step
        out["metadata"] = md
        return out

    if order == "round_robin":
        merged: list[dict[str, Any]] = []
        session_ids = sorted(filtered)
        offsets = {sid: 0 for sid in session_ids}
        while True:
            advanced = False
            for sid in session_ids:
                idx = offsets[sid]
                rows = filtered[sid]
                if idx >= len(rows):
                    continue
                merged.append(_decorate(rows[idx], len(rows), idx + 1))
                offsets[sid] += 1
                advanced = True
            if not advanced:
                break
        return merged

    merged = []
    for sid in sorted(filtered):
        rows = filtered[sid]
        for idx, row in enumerate(rows, start=1):
            merged.append(_decorate(row, len(rows), idx))
    return merged


def _write_split(split: PreparedSplit, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / f"{split.benchmark}_corpus.jsonl"
    qa_path = out_dir / f"{split.benchmark}_qa.jsonl"
    save_jsonl(split.corpus, corpus_path)
    save_jsonl(split.qa, qa_path)
    return corpus_path, qa_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark data files for GapRAG")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["nq", "hotpotqa", "fever", "continual_qa", "all"],
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--summary-path", default="data/processed/benchmark_summary.json")
    parser.add_argument("--nq-limit", type=int, default=500)
    parser.add_argument("--hotpotqa-limit", type=int, default=500)
    parser.add_argument("--fever-limit", type=int, default=500)
    parser.add_argument("--nq-split", default="validation")
    parser.add_argument("--hotpotqa-split", default="validation")
    parser.add_argument("--fever-split", default="validation")
    parser.add_argument("--continual-min-session-size", type=int, default=2)
    parser.add_argument(
        "--continual-order",
        choices=["session_grouped", "round_robin"],
        default="session_grouped",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    need_nq = args.benchmark in {"nq", "continual_qa", "all"}
    need_hotpot = args.benchmark in {"hotpotqa", "continual_qa", "all"}
    need_fever = args.benchmark in {"fever", "continual_qa", "all"}

    prepared: dict[str, PreparedSplit] = {}
    summary: dict[str, Any] = {}

    if need_nq:
        split = _prepare_nq(limit=args.nq_limit, split=args.nq_split)
        corpus_path, qa_path = _write_split(split, out_dir)
        prepared["nq"] = split
        summary["nq"] = {
            "corpus_path": str(corpus_path),
            "qa_path": str(qa_path),
            "num_docs": len(split.corpus),
            "num_qa": len(split.qa),
        }

    if need_hotpot:
        split = _prepare_hotpotqa(limit=args.hotpotqa_limit, split=args.hotpotqa_split)
        corpus_path, qa_path = _write_split(split, out_dir)
        prepared["hotpotqa"] = split
        summary["hotpotqa"] = {
            "corpus_path": str(corpus_path),
            "qa_path": str(qa_path),
            "num_docs": len(split.corpus),
            "num_qa": len(split.qa),
        }

    if need_fever:
        split = _prepare_fever(limit=args.fever_limit, split=args.fever_split)
        corpus_path, qa_path = _write_split(split, out_dir)
        prepared["fever"] = split
        summary["fever"] = {
            "corpus_path": str(corpus_path),
            "qa_path": str(qa_path),
            "num_docs": len(split.corpus),
            "num_qa": len(split.qa),
        }

    if args.benchmark in {"continual_qa", "all"}:
        missing = [name for name in ("nq", "hotpotqa", "fever") if name not in prepared]
        if missing:
            raise RuntimeError(f"Cannot build continual_qa because missing splits: {missing}")
        merged_corpus: dict[str, dict[str, Any]] = {}
        for bench in ("nq", "hotpotqa", "fever"):
            for doc in prepared[bench].corpus:
                merged_corpus[doc["id"]] = doc
        merged_rows = _build_continual_rows(
            [
                ("nq", prepared["nq"].qa),
                ("hotpotqa", prepared["hotpotqa"].qa),
                ("fever", prepared["fever"].qa),
            ],
            min_session_size=args.continual_min_session_size,
            order=args.continual_order,
        )
        split = PreparedSplit(
            benchmark="continual_qa",
            corpus=list(merged_corpus.values()),
            qa=merged_rows,
        )
        corpus_path, qa_path = _write_split(split, out_dir)
        summary["continual_qa"] = {
            "corpus_path": str(corpus_path),
            "qa_path": str(qa_path),
            "num_docs": len(split.corpus),
            "num_qa": len(split.qa),
            "continual_min_session_size": int(args.continual_min_session_size),
            "continual_order": args.continual_order,
        }

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, summary_path)

    print("Prepared benchmark data:")
    for name, info in summary.items():
        print(f"  - {name}: docs={info['num_docs']} qa={info['num_qa']}")
        print(f"      corpus={info['corpus_path']}")
        print(f"      qa={info['qa_path']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
