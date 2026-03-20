from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .doc_encoder import EvidenceEncoder
from .gap_estimator import GapEstimator
from .gap_injector import GapInjector
from .gap_memory import BaseGapMemory, build_memory
from .generator import CausalLMGenerator
from .retriever import DenseRetriever, RetrievedDocument
from .utils import safe_div


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-8)


@dataclass
class PipelineOutput:
    prediction: str
    prediction_confidence: float
    retrieved_docs: list[dict[str, Any]]
    gap_norm: float
    memory_norm: float
    gap_stats: dict[str, Any]
    memory_stats: dict[str, Any]
    hidden_stats: dict[str, Any]
    injector_meta: dict[str, Any]
    elapsed_sec: float


class GapRAGPipeline:
    def __init__(
        self,
        generator: CausalLMGenerator,
        retriever: DenseRetriever,
        evidence_encoder: EvidenceEncoder,
        gap_estimator: GapEstimator,
        gap_injector: GapInjector,
        default_mode: str = "gap_memory_ema",
        top_k: int = 5,
        hidden_layer: int = -1,
        hidden_pooling: str = "last_token",
        evidence_source: str = "rag_hidden",
        memory_cfg: dict[str, Any] | None = None,
        prompt_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.generator = generator
        self.retriever = retriever
        self.evidence_encoder = evidence_encoder
        self.gap_estimator = gap_estimator
        self.gap_injector = gap_injector

        self.default_mode = default_mode
        self.top_k = int(top_k)
        self.hidden_layer = int(hidden_layer)
        self.hidden_pooling = hidden_pooling
        self.evidence_source = str(evidence_source)

        self.memory_cfg = memory_cfg or {}
        self.prompt_cfg = prompt_cfg or {}
        self._session_memories: dict[tuple[str, str], BaseGapMemory] = {}

    def _resolve_injected_memory(
        self,
        run_mode: str,
        gap_vec: np.ndarray,
        previous_memory: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if run_mode == "gap_current":
            return gap_vec, {"read_mode": "current_gap_only", "used_previous_memory": False}

        read_mode = str(self.memory_cfg.get("read_mode", "blend_current"))
        current_gap_weight = float(self.memory_cfg.get("current_gap_weight", 0.75))
        current_gap_weight = min(max(current_gap_weight, 0.0), 1.0)
        cold_start = str(self.memory_cfg.get("cold_start", "current_gap"))

        if previous_memory is None:
            if cold_start == "zero":
                injected = np.zeros_like(gap_vec)
                source = "zero"
            else:
                injected = gap_vec
                source = "current_gap"
            return injected, {
                "read_mode": read_mode,
                "used_previous_memory": False,
                "cold_start": source,
                "current_gap_weight": current_gap_weight,
                "memory_component_norm": 0.0,
                "current_component_norm": float(np.linalg.norm(injected)),
            }

        if read_mode == "memory_only":
            injected = previous_memory
            current_component = np.zeros_like(gap_vec)
            memory_component = previous_memory
        elif read_mode == "memory_or_current_gap":
            injected = previous_memory
            current_component = np.zeros_like(gap_vec)
            memory_component = previous_memory
        else:
            current_component = current_gap_weight * gap_vec
            memory_component = (1.0 - current_gap_weight) * previous_memory
            injected = current_component + memory_component

        return injected.astype(np.float32, copy=False), {
            "read_mode": read_mode,
            "used_previous_memory": True,
            "cold_start": "n/a",
            "current_gap_weight": current_gap_weight,
            "memory_component_norm": float(np.linalg.norm(memory_component)),
            "current_component_norm": float(np.linalg.norm(current_component)),
            "prev_gap_cosine": safe_div(
                float(np.dot(previous_memory, gap_vec)),
                float(np.linalg.norm(previous_memory) * np.linalg.norm(gap_vec)),
            ),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GapRAGPipeline":
        gen_cfg = config.get("generator", {})
        ret_cfg = config.get("retriever", {})
        gap_cfg = config.get("gap", {})
        mem_cfg = config.get("memory", {})
        inj_cfg = config.get("injection", {})
        doc_cfg = config.get("doc_encoder", {})
        pipe_cfg = config.get("pipeline", {})

        generator = CausalLMGenerator(
            model_name=gen_cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            device=gen_cfg.get("device"),
            torch_dtype=gen_cfg.get("torch_dtype", "auto"),
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 128)),
            do_sample=bool(gen_cfg.get("do_sample", False)),
            temperature=float(gen_cfg.get("temperature", 0.0)),
            top_p=float(gen_cfg.get("top_p", 1.0)),
            use_chat_template=bool(gen_cfg.get("use_chat_template", True)),
            system_prompt=gen_cfg.get("system_prompt"),
        )

        retriever = DenseRetriever(
            encoder_name=ret_cfg.get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=ret_cfg.get("device"),
            normalize_embeddings=bool(ret_cfg.get("normalize_embeddings", True)),
            batch_size=int(ret_cfg.get("batch_size", 64)),
        )
        retriever.load_index(
            index_path=ret_cfg["index_path"],
            metadata_path=ret_cfg["metadata_path"],
            embedding_path=ret_cfg.get("embedding_path"),
        )

        evidence_encoder = EvidenceEncoder(method=doc_cfg.get("method", "score_weighted_mean"))
        gap_estimator = GapEstimator(
            gap_type=gap_cfg.get("type", "diff"),
            target_dim=gap_cfg.get("target_dim", generator.hidden_size),
            confidence_weight_source=gap_cfg.get("confidence_weight_source", "retrieval"),
            normalize_inputs=bool(gap_cfg.get("normalize_inputs", True)),
            normalize_gap=bool(gap_cfg.get("normalize_gap", True)),
            max_gap_norm=gap_cfg.get("max_gap_norm", 1.0),
        )
        gap_injector = GapInjector(
            mode=inj_cfg.get("mode", "residual_hidden"),
            alpha=float(inj_cfg.get("alpha", 0.3)),
            prefix_length=int(inj_cfg.get("prefix_length", 1)),
        )

        return cls(
            generator=generator,
            retriever=retriever,
            evidence_encoder=evidence_encoder,
            gap_estimator=gap_estimator,
            gap_injector=gap_injector,
            default_mode=pipe_cfg.get("mode", "gap_memory_ema"),
            top_k=int(ret_cfg.get("top_k", 5)),
            hidden_layer=int(gap_cfg.get("hidden_layer", -1)),
            hidden_pooling=gap_cfg.get("hidden_pooling", "last_token"),
            evidence_source=gap_cfg.get("evidence_source", "rag_hidden"),
            memory_cfg=mem_cfg,
            prompt_cfg=config.get("prompt", {}),
        )

    def _format_vanilla_prompt(self, question: str) -> str:
        template = self.prompt_cfg.get("vanilla_template", "Question: {question}\nAnswer:")
        return template.format(question=question)

    def _format_rag_prompt(self, question: str, docs: list[RetrievedDocument]) -> str:
        context_prefix = self.prompt_cfg.get("context_prefix", "Context:\n")
        context_sep = self.prompt_cfg.get("context_sep", "\n\n")
        qa_template = self.prompt_cfg.get("rag_template", "{context}\nQuestion: {question}\nAnswer:")

        snippets = []
        for i, doc in enumerate(docs, start=1):
            snippets.append(f"[{i}] {doc.text}")
        context = context_prefix + context_sep.join(snippets)
        return qa_template.format(context=context, question=question)

    def _format_context_prompt(self, docs: list[RetrievedDocument]) -> str:
        context_prefix = self.prompt_cfg.get("context_prefix", "Context:\n")
        context_sep = self.prompt_cfg.get("context_sep", "\n\n")
        context_template = self.prompt_cfg.get(
            "context_only_template",
            "{context}\nRespond with only the key evidence span in English.",
        )
        snippets = []
        for i, doc in enumerate(docs, start=1):
            snippets.append(f"[{i}] {doc.text}")
        context = context_prefix + context_sep.join(snippets)
        return context_template.format(context=context)

    def _mode_memory_type(self, mode: str) -> str:
        if mode == "gap_current":
            return "none"
        if mode == "gap_memory_keyed":
            return "keyed"
        if mode == "gap_memory_ema":
            return "ema"
        return str(self.memory_cfg.get("type", "ema"))

    def _get_session_memory(self, mode: str, session_id: str) -> BaseGapMemory:
        mem_type = self._mode_memory_type(mode)
        key = (mem_type, str(session_id))
        if key not in self._session_memories:
            self._session_memories[key] = build_memory(mem_type, memory_cfg=self.memory_cfg)
        return self._session_memories[key]

    def reset_session(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._session_memories = {}
            return
        session_id = str(session_id)
        for key in list(self._session_memories.keys()):
            if key[1] == session_id:
                del self._session_memories[key]

    def run_query(
        self,
        question: str,
        mode: str | None = None,
        session_id: str = "default",
        top_k: int | None = None,
    ) -> PipelineOutput:
        run_mode = mode or self.default_mode
        k = int(top_k or self.top_k)
        k = max(1, k)
        t0 = time.perf_counter()

        if run_mode == "vanilla_lm":
            prompt = self._format_vanilla_prompt(question)
            gen = self.generator.generate(prompt)
            return PipelineOutput(
                prediction=gen.text,
                prediction_confidence=gen.prediction_confidence,
                retrieved_docs=[],
                gap_norm=0.0,
                memory_norm=0.0,
                gap_stats={"type": "none"},
                memory_stats={"type": "none"},
                hidden_stats={},
                injector_meta=gen.injector_meta,
                elapsed_sec=float(time.perf_counter() - t0),
            )

        docs = self.retriever.retrieve(question, top_k=k)
        rag_prompt = self._format_rag_prompt(question, docs)

        if run_mode == "standard_rag":
            gen = self.generator.generate(rag_prompt)
            avg_score = float(np.mean([d.score for d in docs])) if docs else 0.0
            return PipelineOutput(
                prediction=gen.text,
                prediction_confidence=gen.prediction_confidence,
                retrieved_docs=[
                    {
                        "doc_id": d.doc_id,
                        "score": d.score,
                        "text": d.text,
                        "metadata": d.metadata,
                    }
                    for d in docs
                ],
                gap_norm=0.0,
                memory_norm=0.0,
                gap_stats={"type": "none", "avg_retrieval_score": avg_score},
                memory_stats={"type": "none"},
                hidden_stats={},
                injector_meta=gen.injector_meta,
                elapsed_sec=float(time.perf_counter() - t0),
            )

        query_prompt = self._format_vanilla_prompt(question)
        query_hidden = self.generator.extract_hidden(
            query_prompt,
            layer_index=self.hidden_layer,
            pooling=self.hidden_pooling,
        )

        retrieval_scores = np.asarray([d.score for d in docs], dtype=np.float32)
        retrieval_probs = _softmax(retrieval_scores) if retrieval_scores.size else np.zeros(0, dtype=np.float32)
        retrieval_conf = float(np.max(retrieval_probs)) if retrieval_probs.size else 0.0

        if self.evidence_source == "retriever_embedding":
            doc_embeddings = []
            for doc in docs:
                if doc.embedding is not None:
                    doc_embeddings.append(doc.embedding)
                else:
                    doc_embeddings.append(self.retriever.encode_query(doc.text))
            doc_embeddings_np = np.asarray(doc_embeddings, dtype=np.float32)
            query_embed = self.retriever.encode_query(question)
            evidence_vec = self.evidence_encoder.aggregate(
                doc_embeddings=doc_embeddings_np,
                retrieval_scores=retrieval_scores,
                query_embedding=query_embed,
            )
        elif self.evidence_source == "context_hidden":
            context_prompt = self._format_context_prompt(docs)
            evidence_vec = self.generator.extract_hidden(
                context_prompt,
                layer_index=self.hidden_layer,
                pooling=self.hidden_pooling,
            )
        else:
            evidence_vec = self.generator.extract_hidden(
                rag_prompt,
                layer_index=self.hidden_layer,
                pooling=self.hidden_pooling,
            )

        gap_estimate = self.gap_estimator.compute(
            query_vec=query_hidden,
            evidence_vec=evidence_vec,
            retrieval_confidence=retrieval_conf,
        )
        gap_vec = gap_estimate.vector

        memory_type = self._mode_memory_type(run_mode)
        previous_memory: np.ndarray | None = None
        if memory_type == "none":
            memory_vec = gap_vec
            memory_stats = {
                "type": "none",
                "updated": False,
                "prev_norm": 0.0,
                "current_norm": float(np.linalg.norm(gap_vec)),
                "injected_norm": float(np.linalg.norm(gap_vec)),
                "read_mode": "current_gap_only",
                "memory_component_norm": 0.0,
                "current_component_norm": float(np.linalg.norm(gap_vec)),
            }
        else:
            memory = self._get_session_memory(run_mode, session_id)
            previous_memory = memory.get(query_key=query_hidden)
            memory_vec, read_stats = self._resolve_injected_memory(
                run_mode=run_mode,
                gap_vec=gap_vec,
                previous_memory=previous_memory,
            )
            update_min_retrieval_conf = float(self.memory_cfg.get("update_min_retrieval_confidence", 0.0))
            should_update = retrieval_conf >= update_min_retrieval_conf
            memory_stats = {
                "type": memory_type,
                "updated": False,
                "prev_norm": float(np.linalg.norm(previous_memory)) if previous_memory is not None else 0.0,
                "current_norm": float(np.linalg.norm(previous_memory)) if previous_memory is not None else 0.0,
                "injected_norm": float(np.linalg.norm(memory_vec)),
                "update_min_retrieval_confidence": update_min_retrieval_conf,
                **read_stats,
            }

        injection_mode = self.gap_injector.mode
        if run_mode == "gap_current":
            injection_mode = self.gap_injector.mode

        gen = self.generator.generate(
            rag_prompt,
            injector=self.gap_injector,
            gap_vector=memory_vec,
            injection_mode=injection_mode,
            injection_alpha=self.gap_injector.alpha,
            injection_target=self.prompt_cfg.get("injection_target", "last_token"),
            prefix_length=int(self.prompt_cfg.get("prefix_length", 1)),
        )

        if memory_type != "none":
            memory = self._get_session_memory(run_mode, session_id)
            update_min_retrieval_conf = float(self.memory_cfg.get("update_min_retrieval_confidence", 0.0))
            should_update = retrieval_conf >= update_min_retrieval_conf
            if should_update:
                memory.update(gap_vec, query_key=query_hidden)
            post_memory = memory.get(query_key=query_hidden)
            memory_stats["updated"] = bool(should_update)
            memory_stats["current_norm"] = float(np.linalg.norm(post_memory)) if post_memory is not None else 0.0

        return PipelineOutput(
            prediction=gen.text,
            prediction_confidence=gen.prediction_confidence,
            retrieved_docs=[
                {
                    "doc_id": d.doc_id,
                    "score": d.score,
                    "text": d.text,
                    "metadata": d.metadata,
                }
                for d in docs
            ],
            gap_norm=float(np.linalg.norm(gap_vec)),
            memory_norm=float(np.linalg.norm(memory_vec)),
            gap_stats={
                "type": self.gap_estimator.gap_type,
                "evidence_source": self.evidence_source,
                "confidence_weight": gap_estimate.confidence_weight,
                "raw_gap_norm": gap_estimate.raw_gap_norm,
                "avg_retrieval_score": float(np.mean(retrieval_scores)) if retrieval_scores.size else 0.0,
                "retrieval_top_prob": retrieval_conf,
            },
            memory_stats=memory_stats,
            hidden_stats={
                "query_hidden_norm": float(np.linalg.norm(query_hidden)),
                "evidence_hidden_norm": float(np.linalg.norm(evidence_vec)),
                "projected_query_norm": float(np.linalg.norm(gap_estimate.query_proj)),
                "projected_evidence_norm": float(np.linalg.norm(gap_estimate.evidence_proj)),
                "query_evidence_cos": safe_div(
                    float(np.dot(gap_estimate.query_proj, gap_estimate.evidence_proj)),
                    float(np.linalg.norm(gap_estimate.query_proj) * np.linalg.norm(gap_estimate.evidence_proj)),
                ),
            },
            injector_meta=gen.injector_meta,
            elapsed_sec=float(time.perf_counter() - t0),
        )
