from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss-cpu is required for DenseRetriever") from exc

from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    embedding: np.ndarray | None = None


class DenseRetriever:
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 64,
    ) -> None:
        self.encoder_name = encoder_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.index: faiss.Index | None = None
        self.doc_store: list[dict[str, Any]] = []
        self.doc_embeddings: np.ndarray | None = None

    @property
    def embedding_dim(self) -> int:
        if self.doc_embeddings is not None:
            return int(self.doc_embeddings.shape[1])
        return int(self.encoder.get_sentence_embedding_dimension())

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype(np.float32)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.encode_texts([query])
        return vec[0]

    def build_index(
        self,
        corpus: list[dict[str, Any]],
        index_path: str | Path,
        metadata_path: str | Path,
        embedding_path: str | Path,
    ) -> None:
        if not corpus:
            raise ValueError("Corpus is empty")

        texts = [str(doc["text"]) for doc in corpus]
        embeddings = self.encode_texts(texts)

        if self.normalize_embeddings:
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
        else:
            index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        self.doc_store = corpus
        self.doc_embeddings = embeddings

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        embedding_path = Path(embedding_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(index_path))
        np.save(embedding_path, embeddings)
        with metadata_path.open("w", encoding="utf-8") as f:
            for doc in corpus:
                f.write(json.dumps(doc, ensure_ascii=True) + "\n")

    def load_index(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        embedding_path: str | Path | None = None,
    ) -> None:
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.index = faiss.read_index(str(index_path))

        docs: list[dict[str, Any]] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line:
                    docs.append(json.loads(line))
        self.doc_store = docs

        if embedding_path is not None:
            ep = Path(embedding_path)
            if ep.exists():
                self.doc_embeddings = np.load(ep).astype(np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        if self.index is None:
            raise RuntimeError("Retriever index is not loaded")
        if not self.doc_store:
            raise RuntimeError("Retriever document store is empty")

        top_k = max(1, int(top_k))
        query_emb = self.encode_query(query).reshape(1, -1).astype(np.float32)
        if self.normalize_embeddings:
            faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)

        out: list[RetrievedDocument] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.doc_store):
                continue
            doc = self.doc_store[idx]
            emb = None
            if self.doc_embeddings is not None and idx < len(self.doc_embeddings):
                emb = self.doc_embeddings[idx]
            out.append(
                RetrievedDocument(
                    doc_id=str(doc.get("id", idx)),
                    text=str(doc.get("text", "")),
                    score=float(score),
                    metadata=doc.get("metadata", {}),
                    embedding=emb,
                )
            )
        return out
