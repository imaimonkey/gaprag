# GapVerify

GapVerify는 **retrieval-grounded verification을 위한 training-free latent discrepancy control** 연구 코드베이스입니다.

이 저장소의 중심 질문은 단순합니다.
- retrieved evidence와 모델의 내부 belief state 사이에서 의미 있는 latent discrepancy를 추출할 수 있는가?
- 이 discrepancy를 inference-time control signal로 주입하면 verification 정확도를 높일 수 있는가?
- 이 신호는 어떤 verification setting에서 잘 작동하고, 어디서 약해지는가?

## 프로젝트 포지션

GapVerify는 다음 범위에 집중합니다.
- 메인 태스크: `retrieval-grounded verification / fact checking`
- 메인 방법: `gap_current`
- 기준선: `standard_rag`
- 확장/진단: `gap_memory_keyed`, `gap_memory_ema`

핵심 해석은 다음과 같습니다.
- 이 저장소는 **verification decision control**을 연구합니다.
- 핵심 기여 후보는 `current-gap discrepancy injection`입니다.
- memory 계열은 현재 메인 방법이 아니라 보조 진단 축입니다.

참고 문서:
- [Research Scope](/home/kimhj/GapVerify/RESEARCH_REDEFINITION.md)
- [Benchmark Priority](/home/kimhj/GapVerify/BENCHMARK_PRIORITY.md)
- [Restructure Plan](/home/kimhj/GapVerify/RESTRUCTURE_PLAN.md)
- [Experiment Manual](/home/kimhj/GapVerify/EXPERIMENT_MANUAL.md)
- [Related Work Benchmarks](/home/kimhj/GapVerify/RELATED_WORK_BENCHMARKS.md)

## 구현된 모드

- `vanilla_lm`
- `standard_rag`
- `gap_current`
- `gap_memory_ema`
- `gap_memory_keyed`

권장 해석:
- `standard_rag`: verification baseline
- `gap_current`: 메인 실험 방법
- `gap_memory_*`: secondary diagnostic branch

## 지원 벤치마크

메인 verification 벤치:
- `fever`
- `hover`
- `feverous`
- `averitec`

보조 분석 벤치:
- `nq`
- `hotpotqa`

진단 벤치:
- `continual_qa`

권장 역할:
- `fever`, `hover`, `feverous`, `averitec`: verification core result
- `nq`, `hotpotqa`: transfer boundary 분석
- `continual_qa`: memory behavior diagnostic

다음 후보:
- `SciFact`
- `Climate-FEVER`

## 핵심 아이디어

주어진 claim/query와 retrieved evidence에 대해, GapVerify는 다음 두 표현 사이의 discrepancy를 계산합니다.
- query-conditioned hidden representation
- evidence-aligned hidden representation

이 discrepancy를 다시 추론 입력 쪽에 주입하여 verdict formation을 조정합니다.

요약하면:
- retrieval은 evidence를 가져오고
- gap estimator는 latent discrepancy를 만들고
- injector는 그 discrepancy를 training-free control signal로 사용합니다.

## 프로젝트 구조

```text
gapverify/
  README.md
  EXPERIMENT_MANUAL.md
  RELATED_WORK_BENCHMARKS.md
  TODO.md
  RESEARCH_REDEFINITION.md
  BENCHMARK_PRIORITY.md
  RESTRUCTURE_PLAN.md
  pyproject.toml
  requirements.txt

  configs/
    base.yaml
    fever.yaml
    hover.yaml
    feverous.yaml
    averitec.yaml
    nq.yaml
    hotpotqa.yaml
    continual_qa.yaml
    gapverify_current.yaml
    gapverify_memory.yaml
    smoke_tiny.yaml
    standard/
    toptier/

  data/
    raw/
    processed/
    indices/

  scripts/
    prepare_benchmark_data.py
    build_index.py
    run_eval.py
    run_continual_eval.py
    run_ablation.py
    analyze_results.py
    run_experiments.sh
    run_standard.sh
    run_ttv.sh

  gapverify/
    retriever.py
    generator.py
    hidden_extractor.py
    doc_encoder.py
    gap_estimator.py
    gap_memory.py
    gap_injector.py
    pipeline.py
    datasets.py
    metrics.py
    utils.py
    logging_utils.py

  outputs/
    runs/
    tables/
    figures/
```

## 설치

### 옵션 A: pip

```bash
cd /home/kimhj/GapVerify
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

### 옵션 B: uv

```bash
cd /home/kimhj/GapVerify
uv venv
uv sync
```

## 데이터 준비

현재 로컬 adapter가 지원하는 데이터 준비 명령:

```bash
uv run python scripts/prepare_benchmark_data.py --benchmark fever --fever-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark hover --hover-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark feverous --feverous-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark averitec --averitec-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark nq --nq-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark hotpotqa --hotpotqa-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark continual_qa --nq-limit 500 --hotpotqa-limit 500 --fever-limit 500
```

예상 QA 스키마:

```python
{
  "id": "...",
  "question": "...",
  "answers": ["..."],
  "context": ["optional_gold_doc_ids"],
  "metadata": {...},
  "session_id": "..."
}
```

## 인덱스 구축

```bash
uv run python scripts/build_index.py --config configs/fever.yaml
```

## 실행 방법

### 기본 실행

아무 옵션 없이 아래처럼 실행하면 verification core preset이 동작합니다.

```bash
cd /home/kimhj/GapVerify/scripts
sbatch run_experiments.sh
```

기본 preset:
- `BENCHMARK_SUITE=fever,hover,feverous,averitec`
- `MODE_SUITE=standard_rag,gap_current`
- `PREP_BENCHMARK_DATA=true`
- `RUN_BUILD_INDEX=true`
- `RUN_EVAL_STATELESS=true`
- `RUN_EVAL_CONTINUAL=false`

### 표준형 verification 실행

현재 코드 범위에서 stronger local stack을 사용한 표준형 preset:

```bash
cd /home/kimhj/GapVerify/scripts
sbatch run_standard.sh
```

### top-tier 지향 verification 실행

현재 코드 범위에서 더 강한 config variant를 사용한 실험:

```bash
cd /home/kimhj/GapVerify/scripts
sbatch run_ttv.sh
```

주의:
- `run_ttv.sh`는 현재 코드베이스 안에서 가능한 상향 세팅입니다.
- field-standard full stack을 완전히 재현하는 것은 아닙니다.

### 개별 프리셋 실행

```bash
EXPERIMENT_PRESET=verification_core sbatch run_experiments.sh
EXPERIMENT_PRESET=verification_core_standard sbatch run_experiments.sh
EXPERIMENT_PRESET=verification_core_toptier sbatch run_experiments.sh
EXPERIMENT_PRESET=transfer_boundary sbatch run_experiments.sh
EXPERIMENT_PRESET=memory_diagnostic sbatch run_experiments.sh
EXPERIMENT_PRESET=fever_only sbatch run_experiments.sh
```

## 결과 읽기

### stateless run
- 파일: `metrics_summary.json`
- 핵심 값:
  - `exact_match`
  - `f1`
  - `count`

### continual / memory diagnostic
- 파일: `compare_summary.json`
- 핵심 값:
  - `delta_exact_match`
  - `delta_f1`
  - `changed_raw_count`
  - `changed_prediction_count`
  - `improved_count`
  - `regressed_count`

## 현재 문서화된 주장 범위

이 저장소는 다음 질문을 검증하기 위한 연구용 코드입니다.
- latent discrepancy가 verification에서 유효한 control signal인가?
- 그 효과는 benchmark에 따라 얼마나 안정적인가?
- stronger baseline과 config variant로 가면 신호가 유지되는가?

즉 이 README는 현재 `GapVerify` 자체를 기준으로 작성되어 있으며, verification 중심 구조와 실험 축을 기본 전제로 둡니다.
