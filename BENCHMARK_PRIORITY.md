# GapVerify Benchmark Priority

## 1. 목적

이 문서는 GapVerify를 verification 중심 프로젝트로 재정의한 뒤,
어떤 benchmark를 어떤 우선순위로 사용하고 확장할지 정리합니다.

## 2. 우선순위

### Priority 1: 현재 즉시 사용 / 메인 결과

#### FEVER
- 역할: canonical verification benchmark
- 상태: 구현됨
- 목적: `standard_rag` vs `gap_current`의 메인 positive result 확인

#### HoVer
- 역할: many-hop verification benchmark
- 상태: 구현됨
- 목적: evidence composition과 multi-hop verification에서의 성능 확인

#### FEVEROUS
- 역할: structured-evidence flavored verification benchmark
- 상태: 구현됨
- 목적: evidence format이 복합적일 때의 verification 성능 확인

#### AVeriTeC
- 역할: 최신 real-world fact-checking benchmark
- 상태: 구현됨
- 목적: 최신 fact-checking setting으로의 전이 확인

### Priority 2: 다음 통합 대상

#### SciFact
- 역할: scientific claim verification
- 상태: 미구현
- 이유: 도메인 이동성 확인

#### Climate-FEVER
- 역할: climate misinformation verification
- 상태: 미구현
- 이유: domain-specific robustness 확인

## 3. 현재 구현된 벤치의 역할

### FEVER / HoVer / FEVEROUS / AVeriTeC
- 메인 verification benchmark family

### NQ / HotpotQA
- 레거시 free-form QA transfer-boundary benchmark

### continual_qa
- 레거시 persistent memory diagnostic benchmark

## 4. 논문용 최소 세트

### 최소 세트
- `FEVER`
- `AVeriTeC`
- `HoVer`

의미:
- canonical verification
- realistic fact checking
- reasoning-heavy verification

### 권장 세트
- `FEVER`
- `AVeriTeC`
- `HoVer`
- `FEVEROUS`

의미:
- 위 + structured evidence

### 확장 세트
- 권장 세트 + `SciFact` 또는 `Climate-FEVER`

## 5. 현재 repo 기준 즉시 실행 세트

### Main verification
- `fever`
- `hover`
- `feverous`
- `averitec`
- methods: `standard_rag`, `gap_current`

### Transfer boundary
- `nq`, `hotpotqa` (optional legacy)
- methods: `standard_rag`, `gap_current`

### Memory diagnostic
- `continual_qa` (optional legacy)
- methods: `standard_rag`, `gap_current`, `gap_memory_keyed`, `gap_memory_ema`

## 6. 해석 규칙

- verification family에서 positive가 재현되면
  - discrepancy control is verification-oriented

- `NQ/HotpotQA`에서 negative가 나와도
  - answer synthesis로의 negative transfer boundary로 해석

- `continual_qa`가 negative면
  - persistent memory는 현재도 core claim이 아님
