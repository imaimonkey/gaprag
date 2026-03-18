#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaprag.datasets import load_corpus
from gaprag.retriever import DenseRetriever
from gaprag.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense retrieval index for GapRAG")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--corpus-path", default=None)
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--embedding-path", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ret_cfg = cfg.get("retriever", {})
    corpus_cfg = cfg.get("corpus", {})

    corpus_path = args.corpus_path or corpus_cfg.get("path")
    if not corpus_path:
        raise ValueError("corpus.path is missing in config and --corpus-path is not provided")

    index_path = args.index_path or ret_cfg.get("index_path", "data/indices/corpus.faiss")
    metadata_path = args.metadata_path or ret_cfg.get("metadata_path", "data/indices/corpus_metadata.jsonl")
    embedding_path = args.embedding_path or ret_cfg.get("embedding_path", "data/indices/corpus_embeddings.npy")

    corpus = load_corpus(corpus_path)
    print(f"Loaded corpus: {len(corpus)} docs from {corpus_path}")

    retriever = DenseRetriever(
        encoder_name=ret_cfg.get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device=ret_cfg.get("device"),
        normalize_embeddings=bool(ret_cfg.get("normalize_embeddings", True)),
        batch_size=int(ret_cfg.get("batch_size", 64)),
    )

    retriever.build_index(
        corpus=corpus,
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_path=embedding_path,
    )

    print("Index build done")
    print(f"  index     : {Path(index_path)}")
    print(f"  metadata  : {Path(metadata_path)}")
    print(f"  embeddings: {Path(embedding_path)}")


if __name__ == "__main__":
    main()
