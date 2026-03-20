from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .gap_injector import GapInjector
from .hidden_extractor import extract_hidden_vector


@dataclass
class GenerationOutput:
    text: str
    token_count: int
    prediction_confidence: float
    injector_meta: dict[str, Any]


class CausalLMGenerator:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        torch_dtype: str = "auto",
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        use_chat_template: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.use_chat_template = bool(use_chat_template)
        self.system_prompt = None if system_prompt is None else str(system_prompt).strip()

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _encode(self, prompt: str) -> dict[str, torch.Tensor]:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = self._build_messages(prompt)
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                encoded = self.tokenizer(rendered, return_tensors="pt")
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.device) for k, v in encoded.items()}

    def extract_hidden(
        self,
        prompt: str,
        layer_index: int = -1,
        pooling: str = "last_token",
        injector: GapInjector | None = None,
        gap_vector: np.ndarray | None = None,
        injection_mode: str = "none",
        injection_alpha: float = 0.0,
    ) -> np.ndarray:
        encoded = self._encode(prompt)
        if injector is not None and gap_vector is not None and injection_mode != "none":
            payload = injector.prepare(
                self.model,
                encoded,
                gap_vector,
                mode=injection_mode,
                alpha=injection_alpha,
            )
            model_kwargs = payload.model_kwargs
        else:
            model_kwargs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }

        with torch.no_grad():
            outputs = self.model(
                **model_kwargs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden = extract_hidden_vector(
            hidden_states=outputs.hidden_states,
            attention_mask=model_kwargs["attention_mask"],
            layer_index=layer_index,
            pooling=pooling,
        )
        return hidden[0].detach().cpu().float().numpy()

    def generate(
        self,
        prompt: str,
        injector: GapInjector | None = None,
        gap_vector: np.ndarray | None = None,
        injection_mode: str = "none",
        injection_alpha: float = 0.0,
        injection_target: str = "last_token",
        prefix_length: int = 1,
        max_new_tokens: int | None = None,
    ) -> GenerationOutput:
        encoded = self._encode(prompt)
        original_prompt_len = int(encoded["input_ids"].shape[1])
        injector_meta: dict[str, Any] = {"mode": "none", "applied": False}

        if injector is not None and gap_vector is not None and injection_mode != "none":
            payload = injector.prepare(
                self.model,
                encoded,
                gap_vector,
                mode=injection_mode,
                alpha=injection_alpha,
                target=injection_target,
                prefix_length=prefix_length,
            )
            model_inputs = payload.model_kwargs
            injector_meta = payload.meta
        else:
            model_inputs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": self.do_sample,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        with torch.no_grad():
            generation = self.model.generate(**model_inputs, **generate_kwargs)

        sequences = generation.sequences
        if sequences.shape[1] > original_prompt_len:
            generated_ids = sequences[:, original_prompt_len:]
        else:
            generated_ids = sequences

        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        token_count = int(generated_ids.shape[1])

        confidence = 0.0
        if generation.scores:
            probs = []
            for step_scores in generation.scores:
                step_prob = torch.softmax(step_scores, dim=-1).max(dim=-1).values.mean().item()
                probs.append(step_prob)
            confidence = float(np.mean(probs)) if probs else 0.0

        return GenerationOutput(
            text=text,
            token_count=token_count,
            prediction_confidence=confidence,
            injector_meta=injector_meta,
        )
