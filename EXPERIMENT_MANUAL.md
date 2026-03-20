# GapVerify 실험 매뉴얼

## 1. 이 프로젝트를 어떻게 읽을 것인가

GapVerify의 기본 단위는 아래입니다.
- 태스크: `retrieval-grounded verification`
- 기준선: `standard_rag`
- 메인 방법: `gap_current`
- 보조 축: `gap_memory_keyed`, `gap_memory_ema`

실험은 세 층으로 나뉩니다.
1. verification core
2. transfer boundary
3. memory diagnostic

## 2. 환경 준비

```bash
cd /home/kimhj/GapVerify
uv venv
uv sync --extra dev
```

## 3. 데이터 준비

### 3.1 지원 벤치

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

### 3.2 수동 준비 예시

```bash
uv run python scripts/prepare_benchmark_data.py --benchmark fever --fever-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark hover --hover-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark feverous --feverous-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark averitec --averitec-limit 500
```

## 4. 실행 프리셋

### 4.1 기본 실행

```bash
cd /home/kimhj/GapVerify/scripts
sbatch run_experiments.sh
```

이 명령은 `verification_core` preset을 실행합니다.
- 벤치: `fever,hover,feverous,averitec`
- 모드: `standard_rag,gap_current`
- 목적: verification 계열에서의 기본 효과 확인

### 4.2 표준형 실행

```bash
cd /home/kimhj/GapVerify/scripts
sbatch run_standard.sh
```

이 preset은 현재 코드 범위에서 baseline stack을 강화합니다.
- stronger retriever config
- 별도 standard index
- standard variant config 사용

용도:
- 기본 세팅에서 보인 signal이 조금 더 강한 baseline에서도 유지되는지 확인

### 4.3 top-tier 지향 실행

```bash
cd /home/kimhj/GapVerify/scripts
sbatch run_ttv.sh
```

용도:
- 현재 코드 안에서 가능한 상향 세팅으로 verification core 재평가

주의:
- 이것은 field-standard full system 재현이 아니라, 현재 repo 안에서의 stronger variant입니다.

### 4.4 transfer boundary 실행

```bash
cd /home/kimhj/GapVerify/scripts
EXPERIMENT_PRESET=transfer_boundary sbatch run_experiments.sh
```

대상:
- `nq`
- `hotpotqa`

질문:
- verification에서 작동한 discrepancy control이 free-form QA에도 전이되는가?

### 4.5 memory diagnostic 실행

```bash
cd /home/kimhj/GapVerify/scripts
EXPERIMENT_PRESET=memory_diagnostic sbatch run_experiments.sh
```

대상:
- `continual_qa`

질문:
- memory가 output을 얼마나 바꾸는가?
- 변화가 실제 gain으로 이어지는가?

## 5. 결과 파일 읽기

### 5.1 stateless benchmark 결과
- 파일: `outputs/runs/<run_name>/metrics_summary.json`
- 핵심 값:
  - `count`
  - `exact_match`
  - `f1`

### 5.2 continual/memory 결과
- 파일: `outputs/runs/<run_name>/compare_summary.json`
- 핵심 값:
  - `delta_exact_match`
  - `delta_f1`
  - `changed_raw_count`
  - `changed_prediction_count`
  - `improved_count`
  - `regressed_count`

### 5.3 해석 원칙

- `standard_rag -> gap_current`가 오르면:
  - discrepancy injection이 해당 verification setting에서 유효하다는 뜻입니다.

- `changed_raw_count > 0`인데 `delta_exact_match <= 0`이면:
  - injector는 active하지만 품질 개선에는 실패한 것입니다.

- stronger preset에서 baseline이 오르고 `gap_current`가 줄거나 역전되면:
  - 현재 방법의 signal이 약한 baseline 보정 효과인지, robust한 additive gain인지 다시 분해해야 합니다.

## 6. 현재 권장 실험 순서

1. `verification_core`
2. `verification_core_standard`
3. `verification_core_toptier`
4. `transfer_boundary`
5. `memory_diagnostic`

즉 우선순위는 항상 verification 계열입니다.

## 7. 현재 문서 기준 주장 범위

문서 기준으로 메인 서사는 아래입니다.
- GapVerify는 verification decision control을 위한 연구 코드다.
- 메인 실험 단위는 `standard_rag`와 `gap_current` 비교다.
- memory 계열은 secondary branch다.
- QA 벤치는 메인 성능 벤치가 아니라 transfer boundary 분석 축이다.
