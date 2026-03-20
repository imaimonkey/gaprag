# GapVerify 실험 매뉴얼

## 0. 현재 프로젝트를 어떻게 읽어야 하나

이 저장소는 지금 시점에서 아래처럼 읽는 것이 맞습니다.

- 메인 태스크: `retrieval-grounded verification`
- 메인 방법: `gap_current`
- 기준선: `standard_rag`
- 보조/진단: `gap_memory_keyed`, `gap_memory_ema`

즉 현재 실험은
- `persistent memory가 continual gain을 준다`
를 증명하는 단계가 아니라,
- `latent discrepancy가 verification decision control signal로 유효한가`
를 검증하는 단계입니다.

현재 지원 벤치 역할:
- `fever`, `hover`, `feverous`, `averitec`: 메인 verification 벤치군
- `nq`, `hotpotqa`: 레거시 자유생성 QA transfer-boundary 확인
- `continual_qa`: 레거시 persistent memory 진단용

## 1. 환경 준비

```bash
cd /home/kimhj/GapVerify
uv venv
uv sync --extra dev
```

의존성 잠금은 이미 `uv.lock`에 반영되어 있습니다.

## 2. 데이터 준비

### 2.1 현재 repo에서 바로 준비 가능한 벤치

```bash
uv run python scripts/prepare_benchmark_data.py --benchmark fever --fever-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark hover --hover-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark feverous --feverous-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark averitec --averitec-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark nq --nq-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark hotpotqa --hotpotqa-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark continual_qa --nq-limit 500 --hotpotqa-limit 500 --fever-limit 500
```

### 2.2 현재 구현 범위와 다음 목표를 분리해서 보기

현재 구현됨:
- `FEVER`
- `HoVer`
- `FEVEROUS`
- `AVeriTeC`
- `NQ`
- `HotpotQA`
- `continual_qa`

다음 통합 우선순위:
1. `SciFact`
2. `Climate-FEVER`

## 3. 실험 우선순위

### 3.0 기본 실행

이제 기본 명령은 아래 하나입니다.

```bash
cd /home/kimhj/GapVerify/scripts
sbatch run_experiments.sh
```

이 명령은 자동으로 아래 preset을 실행합니다.
- `EXPERIMENT_PRESET=verification_core`
- `fever, hover, feverous, averitec`
- `standard_rag, gap_current`

즉 별도 env를 안 줘도 verification core 실험이 바로 돌도록 바뀌었습니다.

### 3.1 1순위: verification core 4종에서 효과 확인

```bash
cd /home/kimhj/GapVerify
BENCHMARK_SUITE=fever,hover,feverous,averitec \
MODE_SUITE=standard_rag,gap_current \
RUN_NAME_PREFIX=run_verification_core \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
RUN_EVAL_STATELESS=true \
RUN_EVAL_CONTINUAL=false \
sbatch scripts/run_experiments.sh
```

이 실험의 질문:
- `gap_current`가 verification-family benchmark 전반에서 유효한가?

반드시 확인할 파일:
- `outputs/runs/run_verification_core_fever_standard_rag/metrics_summary.json`
- `outputs/runs/run_verification_core_fever_gap_current/metrics_summary.json`
- `outputs/runs/run_verification_core_hover_standard_rag/metrics_summary.json`
- `outputs/runs/run_verification_core_hover_gap_current/metrics_summary.json`
- `outputs/runs/run_verification_core_feverous_standard_rag/metrics_summary.json`
- `outputs/runs/run_verification_core_feverous_gap_current/metrics_summary.json`
- `outputs/runs/run_verification_core_averitec_standard_rag/metrics_summary.json`
- `outputs/runs/run_verification_core_averitec_gap_current/metrics_summary.json`

### 3.2 2순위: 자유생성 QA에서 negative transfer 확인

```bash
cd /home/kimhj/GapVerify
BENCHMARK_SUITE=nq,hotpotqa \
MODE_SUITE=standard_rag,gap_current \
RUN_NAME_PREFIX=run_transfer_boundary \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
RUN_EVAL_CONTINUAL=auto \
sbatch scripts/run_experiments.sh
```

이 실험의 질문:
- verification에서 유효한 discrepancy injection이 free-form QA에도 전이되는가?

해석 원칙:
- 좋아지지 않아도 실패로만 읽지 말고,
- `task boundary`를 보여주는 결과로 읽습니다.

### 3.3 3순위: persistent memory는 진단용으로만 본다

```bash
cd /home/kimhj/GapVerify
BENCHMARK_PROFILE=continual_qa \
MODE_SUITE=standard_rag,gap_current,gap_memory_keyed,gap_memory_ema \
RUN_NAME=run_continual_suite \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
sbatch scripts/run_experiments.sh
```

짧은 preset 실행 예시:

```bash
EXPERIMENT_PRESET=verification_core sbatch run_experiments.sh
EXPERIMENT_PRESET=transfer_boundary sbatch run_experiments.sh
EXPERIMENT_PRESET=memory_diagnostic sbatch run_experiments.sh
EXPERIMENT_PRESET=fever_only sbatch run_experiments.sh
```

이 실험의 질문:
- memory가 실제로 output을 바꾸는가?
- 바꾼다면 accuracy gain으로 이어지는가?

핵심 확인 파일:
- `compare_summary.json`
- `predictions_stateless.jsonl`
- `predictions_continual.jsonl`

핵심 해석 지표:
- `delta_exact_match`
- `changed_raw_count`
- `changed_prediction_count`
- `improved_count`
- `regressed_count`

## 4. 로그와 결과를 어떻게 읽을 것인가

### 4.1 stateless benchmark
- 파일: `metrics_summary.json`
- 핵심 값:
  - `exact_match`
  - `f1`
  - `retrieval_hit_at_k`
  - `avg_prediction_confidence`

### 4.2 continual diagnostic
- 파일: `compare_summary.json`
- 핵심 값:
  - `delta_exact_match`
  - `delta_f1`
  - `changed_raw_count`
  - `improved_count`
  - `regressed_count`

### 4.3 중요한 해석 규칙

- `changed_raw_count > 0`인데 `delta_exact_match <= 0`
  - memory/injection은 행동적으로는 active하지만 품질 향상에는 실패한 것입니다.

- `retrieval_hit_at_k`는 같은데 `exact_match`만 변함
  - retrieval이 아니라 generator-side control 효과입니다.

- `FEVER`에서 좋아지고 `NQ/HotpotQA`에서 악화
  - 이 방법은 answer synthesis보다 verification decision에 맞는다는 뜻입니다.

## 5. 지금 주장하면 안 되는 것

현재 상태에서 아래 주장은 금지하는 게 맞습니다.

- `persistent memory가 continual RAG를 robust하게 개선한다`
- `gap_current가 범용 QA 개선 방법이다`
- `현재 코드가 이미 최신 verification benchmark 전체를 커버한다`

대신 아래처럼 말해야 합니다.

- `latent discrepancy is promising as a control signal for verification-style tasks`
- `the same signal does not currently transfer cleanly to free-form QA`
- `persistent memory remains a negative or unresolved branch`

## 6. 현재 실험 메뉴의 권장 순서

1. `fever`, `hover`, `feverous`, `averitec`: `standard_rag` vs `gap_current`
2. `nq`, `hotpotqa`: 필요할 때만 `standard_rag` vs `gap_current`
3. `continual_qa`: 필요할 때만 `standard_rag`, `gap_current`, `gap_memory_keyed`, `gap_memory_ema`

## 7. 다음 개발 우선순위

1. verification 전용 calibration / confidence analysis 강화
2. current-gap 전용 injector 정교화
3. `SciFact` adapter 추가
4. `Climate-FEVER` adapter 추가
5. persistent memory는 appendix/negative-result 경로로 유지
