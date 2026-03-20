# Related Work Benchmarks

This document tracks recent verification/fact-checking benchmark references and provides a side-by-side template for comparing `GapVerify` against external systems.

## Comparison Rules

- Do not compare raw numbers across benchmarks without matching the metric.
- `FEVER`, `HoVer`, and `FEVEROUS` are usually reported with label accuracy or macro-F1.
- `AVeriTeC` uses the official `AVeriTeC score`; local `exact_match` is not directly comparable.
- When using local `GapVerify` numbers below, always cite the run directory and dataset count.

## Recent External Reference Points

### FEVER

| Method | Year | Metric | Score | Notes | Source |
|---|---:|---|---:|---|---|
| SFAVEL | 2024 | Label Accuracy / FEVER Score | 89.48 / 86.15 | ICLR 2024; official FEVER test | [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/238c98450b1d9e8055f94d22f303bb57-Paper-Conference.pdf) |
| Concept-HGN | 2025 | Label Accuracy / FEVER Score | 80.26 / 77.68 | FEVER results reported in Neural Networks paper | [ScienceDirect abstract](https://www.sciencedirect.com/science/article/pii/S0893608025008408) |
| ProoFVer | 2021 | Label Accuracy / FEVER Score | 79.47 / 76.82 | Still a useful classical reference | [arXiv abstract](https://arxiv.org/abs/2108.11357) |

### HoVer

| Method | Year | Metric | Score | Notes | Source |
|---|---:|---|---:|---|---|
| PACAR | 2024 | Macro-F1 (2/3/4-hop) | 73.13 / 64.07 / 63.82 | Open-book setting | [ACL Anthology PDF](https://aclanthology.org/2024.lrec-main.1099.pdf) |
| PACAR | 2024 | Macro-F1 (2/3/4-hop) | 76.86 / 70.10 / 69.95 | Gold-evidence setting | [ACL Anthology PDF](https://aclanthology.org/2024.lrec-main.1099.pdf) |
| DP-GraphCheck | 2025 | Accuracy (2/3/4-hop) | 77.1 / 73.7 / 73.6 | Open-book + gold setting; paper notes accuracy for consistency | [ACL Anthology page](https://aclanthology.org/2025.findings-emnlp.1345/) |
| DP-GraphCheck | 2025 | Accuracy (2/3/4-hop) | 81.7 / 73.8 / 74.1 | Gold setting | [ACL Anthology page](https://aclanthology.org/2025.findings-emnlp.1345/) |
| ZeFaV | 2024 | Result reported qualitatively | comparable to SOTA | Use only if exact table is extracted later | [arXiv abstract](https://arxiv.org/abs/2411.11247) |

### FEVEROUS
| Method | Year | Metric | Score | Notes | Source |
|---|---:|---|---:|---|---|
| PACAR | 2024 | Macro-F1 | 72.61 | Open-book setting | [ACL Anthology PDF](https://aclanthology.org/2024.lrec-main.1099.pdf) |
| PACAR | 2024 | Macro-F1 | 94.43 | Gold-evidence setting | [ACL Anthology PDF](https://aclanthology.org/2024.lrec-main.1099.pdf) |
| DP-GraphCheck | 2025 | Macro-F1 | 66.34 | FEVEROUS-Intro | [ACL Anthology page](https://aclanthology.org/2025.findings-emnlp.1345/) |
| DP-GraphCheck | 2025 | Macro-F1 | 86.87 | FEVEROUS-Alpha | [ACL Anthology page](https://aclanthology.org/2025.findings-emnlp.1345/) |
| ZeFaV | 2024 | Result reported qualitatively | comparable to SOTA | Use only if exact table is extracted later | [arXiv abstract](https://arxiv.org/abs/2411.11247) |

### AVeriTeC

| Method | Year | Metric | Score | Notes | Source |
|---|---:|---|---:|---|---|
| CTU AIC | 2025 | AVeriTeC score | 0.3317 | Winner of 2025 shared task | [AVeriTeC 2025 shared task](https://aclanthology.org/2025.fever-1.15/) |
| Baseline | 2025 | AVeriTeC score | 0.2023 | Official baseline | [AVeriTeC 2025 shared task](https://aclanthology.org/2025.fever-1.15/) |
| CTU AIC (4o-mini) | 2025 | AVeriTeC score | 0.3176 | Strong open/reproducible reference | [AVeriTeC 2025 shared task](https://aclanthology.org/2025.fever-1.15/) |
| HerO | 2024 | AVeriTeC score | 0.57 | 2nd place on FEVER 2024 AVeriTeC | [ACL Anthology page](https://aclanthology.org/2024.fever-1.15/) |
| CTU AIC simple RAG | 2024 | AVeriTeC score | 0.504 | Reframed as simple RAG task | [GitHub summary with paper link](https://github.com/aic-factcheck/aic_averitec) |
| Papelo | 2024 | AVeriTeC score | 0.477 | Shared-task system description | [AVeriTeC shared task overview](https://aclanthology.org/2024.fever-1.1/) |

## Current Local GapVerify Results

Only use rows marked `validated=yes` as current local references.

| Benchmark | Local metric | Count | standard_rag | gap_current | Delta | Status | Run files |
|---|---|---:|---:|---:|---:|---|---|
| FEVER | exact_match / f1 | 498 | 0.3635 | 0.5803 | +0.2169 | validated=yes | [std](/home/kimhj/GapVerify/outputs/runs/run_verification_core_fever_full_standard_rag/metrics_summary.json), [gap](/home/kimhj/GapVerify/outputs/runs/run_verification_core_fever_full_gap_current/metrics_summary.json) |
| HoVer | exact_match / f1 | 500 | 0.4700 | 0.5740 | +0.1040 | validated=yes | [std](/home/kimhj/GapVerify/outputs/runs/run_verification_core_hover_standard_rag/metrics_summary.json), [gap](/home/kimhj/GapVerify/outputs/runs/run_verification_core_hover_gap_current/metrics_summary.json) |
| FEVEROUS | exact_match / f1 | 493 | 0.4990 | 0.5822 | +0.0832 | validated=yes | [std](/home/kimhj/GapVerify/outputs/runs/run_verification_core_feverous_standard_rag/metrics_summary.json), [gap](/home/kimhj/GapVerify/outputs/runs/run_verification_core_feverous_gap_current/metrics_summary.json) |
| AVeriTeC | exact_match / f1 | 500 | 0.6660 | 0.6040 | -0.0620 | validated=yes | [std](/home/kimhj/GapVerify/outputs/runs/run_verification_core_averitec_standard_rag/metrics_summary.json), [gap](/home/kimhj/GapVerify/outputs/runs/run_verification_core_averitec_gap_current/metrics_summary.json) |

## Side-by-Side Comparison Table

Use this table in notes, README drafts, or papers after matching metric and setting.

| Benchmark | Metric | Local standard_rag | Local gap_current | Local delta | Best external reference | External score | Directly comparable? | Notes |
|---|---|---:|---:|---:|---|---:|---|---|
| FEVER | Label accuracy-like verdict metric | 0.3635 | 0.5803 | +0.2169 | SFAVEL (2024) | 89.48 LA / 86.15 FS | partial | Local metric is EM over labels, not official FEVER score |
| HoVer | Verdict metric / macro-F1 family | 0.4700 | 0.5740 | +0.1040 | PACAR (2024) | 73.13 / 64.07 / 63.82 macro-F1 | partial | Need hop-wise breakdown for fair comparison |
| FEVEROUS | Verdict metric / macro-F1 family | 0.4990 | 0.5822 | +0.0832 | PACAR (2024) | 72.61 macro-F1 | partial | Need exact evidence/text setting alignment |
| AVeriTeC | Local EM vs official AVeriTeC score | 0.6660 | 0.6040 | -0.0620 | CTU AIC (2025) | 0.3317 | no | Local EM is not the official AVeriTeC score; do not compare directly |

## Blank Template

Copy and fill this as new runs finish.

| Benchmark | Setting | Metric | Local run dir | standard_rag | gap_current | gap_memory_keyed | gap_memory_ema | Best external method | External metric | External score | Comparable? | Note |
|---|---|---|---|---:|---:|---:|---:|---|---|---:|---|---|
| FEVER | full / stateless |  |  |  |  |  |  |  |  |  |  |  |
| HoVer | full / stateless |  |  |  |  |  |  |  |  |  |  |  |
| FEVEROUS | full / stateless |  |  |  |  |  |  |  |  |  |  |  |
| AVeriTeC | full / stateless |  |  |  |  |  |  |  |  |  |  |  |

## Immediate Next Step

- Keep FEVER, HoVer, and FEVEROUS as current local anchors.
- Keep `AVeriTeC` as a completed local run, but treat it as diagnostic only.
- Add the official `AVeriTeC score` evaluator before claiming any direct comparison against shared-task systems.
