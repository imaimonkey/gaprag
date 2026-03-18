# GapRAG TODO

## Near-term

- Add query-aware confidence from generation entropy into `confidence_weighted` gap.
- Add explicit retrieval confidence calibration and uncertainty logging.
- Add layer sweep runner for early/mid/late hidden extraction ablation.
- Add optional token-level aligned gap prototype.

## Mid-term

- Implement attention-bias injection backend for selected causal LMs.
- Add larger benchmark adapters (NQ / TriviaQA / Hotpot-style / PopQA).
- Add conflict-heavy continual sessions and domain shift scenarios.
- Add richer hallucination faithfulness checks with evidence attribution.

## Engineering

- Add unit tests for each module (`gap_estimator`, `gap_memory`, `gap_injector`, `pipeline`).
- Add CI for lint + py_compile + smoke checks.
- Add run registry utility for tabular result aggregation across runs.
