# GapVerify

GapVerify는 **retrieval-grounded verification에서의 latent discrepancy control**을 연구하기 위한 모듈형 코드베이스입니다.

이 코드베이스는 원래 continual RAG를 위한 persistent latent discrepancy memory를 탐색하는 방향에서 시작했습니다. 그러나 현재까지의 실험 결과를 종합하면, 더 좁고 방어 가능한 현재 위치는 다음과 같습니다.
- `gap_current`가 현재의 핵심 방법입니다.
- 주된 태스크 계열은 evidence-grounded **verification**입니다.
- persistent memory(`gap_memory_ema`, `gap_memory_keyed`)는 현재 핵심 주장이라기보다 진단용/negative-result 분기입니다.

## 현재 연구 포지션

이 저장소에서 현재 경험적으로 지지되는 결론은 다음과 같습니다.
- `FEVER`: latent discrepancy injection이 label-style verification에 도움이 될 수 있습니다.
- `HoVer`, `FEVEROUS`, `AVeriTeC`: 현재 이 저장소에 verification 계열 벤치마크로 통합되어 있습니다.
- `NQ`, `HotpotQA`: 동일한 injection이 free-form QA에는 **안정적으로 전이되지 않습니다**.
- `continual_qa`: persistent memory는 현재 긍정적인 continual gain을 보여주지 못합니다.

따라서 이 저장소는 다음과 같이 읽는 것이 맞습니다.
- **핵심 질문**: model-evidence latent discrepancy가 training-free control signal로 작동할 수 있는가?
- **주요 태스크**: retrieval-grounded verification / fact checking
- **부차 질문**: 이 신호는 어디에서 전이에 실패하는가?

참고 문서:
- [Research Redefinition](/home/kimhj/GapVerify/RESEARCH_REDEFINITION.md)
- [Benchmark Priority](/home/kimhj/GapVerify/BENCHMARK_PRIORITY.md)
- [Restructure Plan](/home/kimhj/GapVerify/RESTRUCTURE_PLAN.md)
- [Experiment Manual](/home/kimhj/GapVerify/EXPERIMENT_MANUAL.md)

## 구현된 모드

- `vanilla_lm`
- `standard_rag`
- `gap_current`
- `gap_memory_ema`
- `gap_memory_keyed`

해석:
- `standard_rag`: baseline
- `gap_current`: 현재의 주 실험 방법
- `gap_memory_*`: 보조 진단용 방법

## 이 저장소에서 현재 지원하는 벤치마크

현재 구현됨:
- `fever`
- `hover`
- `feverous`
- `averitec`
- `nq`
- `hotpotqa`
- `continual_qa`

권장 역할:
- `fever`, `hover`, `feverous`, `averitec`: 메인 verification 벤치마크 계열
- `nq`, `hotpotqa`: 레거시 transfer-boundary / negative-transfer 분석
- `continual_qa`: 레거시 memory diagnostic 벤치마크

다음 verification 벤치 후보:
- optional: `SciFact`, `Climate-FEVER`

## 핵심 아이디어

주어진 query/claim과 retrieved evidence에 대해, GapVerify는 다음 두 상태 사이의 latent discrepancy를 추정합니다.
- 모델의 query-conditioned hidden state
- evidence-aligned hidden representation

이 discrepancy를 다시 추론 과정에 주입하여 training-free control signal로 사용합니다.

현재 근거를 갖고 주장할 수 있는 내용은 다음입니다.
- 이 신호는 **verification-style label decision**에 도움이 될 수 있습니다.
- 하지만 free-form QA generation을 안정적으로 개선하는 방법으로는 아직 보이지 않습니다.

## 프로젝트 구조

```text
gapverify/
  README.md
  EXPERIMENT_MANUAL.md
  TODO.md
  RESEARCH_REDEFINITION.md
  BENCHMARK_PRIORITY.md
  RESTRUCTURE_PLAN.md
  pyproject.toml
  requirements.txt

  configs/
    base.yaml
    nq.yaml
    hotpotqa.yaml
    fever.yaml
    hover.yaml
    feverous.yaml
    averitec.yaml
    continual_qa.yaml
    rag.yaml
    gapverify_current.yaml
    gapverify_memory.yaml
    ablation_gap_defs.yaml
    ablation_memory.yaml
    ablation_injection.yaml
    smoke_tiny.yaml

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

현재 로컬 benchmark adapter가 준비하는 대상:
- `fever`
- `hover`
- `feverous`
- `averitec`
- `nq`
- `hotpotqa`
- `continual_qa`

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

## Retrieval Index 구축

```bash
uv run python scripts/build_index.py --config configs/base.yaml
```

## 권장 실험 메뉴

### 0. 기본 1줄 실행

아무 옵션 없이 아래처럼 실행하면:

```bash
cd /home/kimhj/GapVerify/scripts
sbatch run_experiments.sh
```

스크립트는 현재 기본적으로 다음 설정으로 동작합니다.
- `EXPERIMENT_PRESET=verification_core`
- `BENCHMARK_SUITE=fever,hover,feverous,averitec`
- `MODE_SUITE=standard_rag,gap_current`
- `PREP_BENCHMARK_DATA=true`
- `RUN_BUILD_INDEX=true`
- `RUN_EVAL_STATELESS=true`
- `RUN_EVAL_CONTINUAL=false`

### 1. 메인 verification 실행

현재 이 저장소에서 권장하는 verification-core 실행:

```bash
BENCHMARK_SUITE=fever,hover,feverous,averitec \
MODE_SUITE=standard_rag,gap_current \
RUN_NAME_PREFIX=run_verification_core \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
RUN_EVAL_STATELESS=true \
RUN_EVAL_CONTINUAL=false \
sbatch scripts/run_experiments.sh
```

### 2. Transfer-boundary 실행

```bash
BENCHMARK_SUITE=nq,hotpotqa \
MODE_SUITE=standard_rag,gap_current \
RUN_NAME_PREFIX=run_transfer_boundary \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
RUN_EVAL_CONTINUAL=auto \
sbatch scripts/run_experiments.sh
```

### 3. Memory diagnostic 실행

```bash
BENCHMARK_PROFILE=continual_qa \
MODE_SUITE=standard_rag,gap_current,gap_memory_keyed,gap_memory_ema \
RUN_NAME=run_continual_suite \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
sbatch scripts/run_experiments.sh
```

해석:
- `fever`, `hover`, `feverous`, `averitec`를 메인 verification 계열로 사용
- `nq`, `hotpotqa`는 task-boundary failure case로만 사용
- `continual_qa`는 persistent memory 주장을 과장하지 않기 위한 진단용으로만 사용

### Preset 단축 실행

```bash
EXPERIMENT_PRESET=verification_core sbatch scripts/run_experiments.sh
EXPERIMENT_PRESET=transfer_boundary sbatch scripts/run_experiments.sh
EXPERIMENT_PRESET=memory_diagnostic sbatch scripts/run_experiments.sh
EXPERIMENT_PRESET=fever_only sbatch scripts/run_experiments.sh
```

`EXPERIMENT_PRESET=custom`은 `BENCHMARK_PROFILE`, `BENCHMARK_SUITE`, `MODE`, `MODE_SUITE`를 직접 전부 지정하고 싶을 때만 사용하면 됩니다.

## Baseline / Method 개별 실행

### Stateless

```bash
python scripts/run_eval.py --config configs/fever.yaml --mode standard_rag --stateless --run-name fever_rag
python scripts/run_eval.py --config configs/fever.yaml --mode gap_current --stateless --run-name fever_gap_current
```

### Continual diagnostic

```bash
python scripts/run_continual_eval.py --config configs/continual_qa.yaml --mode gap_memory_ema --run-name continual_gap_memory_ema
```

## 결과 분석

```bash
python scripts/analyze_results.py --runs-dir outputs/runs --out-dir outputs/figures
```

현재 권장 해석 방식:
- `compare_summary.json`: continual diagnostic 비교에 사용
- `metrics_summary.json`: stateless benchmark 비교에 사용
- `changed_raw_count`, `changed_prediction_count`: "출력이 바뀌었는가"와 "정확도가 좋아졌는가"를 분리해서 볼 때 사용

## Slurm 배치 실행

주 실행 스크립트:

```bash
sbatch scripts/run_experiments.sh
```

주요 환경변수 옵션:
- `BENCHMARK_PROFILE` (`demo|nq|hotpotqa|fever|continual_qa`)
- `BENCHMARK_SUITE` (콤마 구분 profile sweep)
- `MODE` (단일 모드)
- `MODE_SUITE` (콤마 구분 mode sweep)
- `RUN_NAME`, `RUN_NAME_PREFIX`
- `PREP_BENCHMARK_DATA`
- `RUN_BUILD_INDEX`
- `RUN_EVAL_STATELESS`
- `RUN_EVAL_CONTINUAL`
- `RUN_ABLATION`

`RUN_EVAL_CONTINUAL=auto`의 의미:
- `demo`와 `continual_qa`에서만 continual 비교를 수행
- continual test로 의미 없는 벤치에서 과장된 해석을 막기 위한 장치

## 이 저장소가 현재 주장하지 않는 것

이 코드베이스는 현재 아래의 강한 주장들을 **지지하지 않습니다**.
- persistent memory가 미래 query를 robust한 continual setting에서 개선한다
- discrepancy injection이 범용적인 QA 개선 기법이다
- 현재 벤치 지원만으로 최신 fact-checking suite 전반을 모두 커버한다

이들은 현재 확립된 결론이 아니라, 앞으로 더 검증해야 할 열린 질문 또는 후속 과제입니다.
