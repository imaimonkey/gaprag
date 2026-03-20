# GapVerify Benchmark Priority

## 1. 목적

이 문서는 GapVerify를 verification 중심 프로젝트로 재정의한 뒤,
어떤 benchmark를 어떤 우선순위로 통합하고 사용할지 정리합니다.

## 2. 우선순위

### Priority 1: 현재 즉시 사용 / 메인 결과

#### FEVER
- 역할: 현재 repo에서 가장 중요한 verification benchmark
- 상태: 구현됨
- 목적: `standard_rag` vs `gap_current`의 메인 positive result 확인
- 주요 지표: accuracy / F1 / retrieval_hit_at_k

### Priority 2: 다음 통합 대상

#### AVeriTeC
- 역할: 최신 real-world fact-checking benchmark
- 상태: 미구현
- 이유: 실제 웹 evidence와 fact-checking setting을 반영

#### HoVer
- 역할: many-hop verification benchmark
- 상태: 미구현
- 이유: reasoning/evidence composition 검증

#### FEVEROUS
- 역할: structured evidence benchmark
- 상태: 미구현
- 이유: table/list/cell evidence까지 포함한 verification 확장

### Priority 3: 도메인 확장

#### SciFact
- 역할: scientific claim verification
- 상태: 미구현
- 이유: 도메인 이동성 확인

#### Climate-FEVER
- 역할: climate misinformation verification
- 상태: 미구현
- 이유: domain-specific robustness 확인

## 3. 현재 구현된 벤치의 역할

### FEVER
- 메인 verification benchmark

### NQ
- free-form QA transfer-boundary benchmark

### HotpotQA
- multi-hop free-form QA transfer-boundary benchmark

### continual_qa
- persistent memory diagnostic benchmark

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
- methods: `standard_rag`, `gap_current`

### Transfer boundary
- `nq`, `hotpotqa`
- methods: `standard_rag`, `gap_current`

### Memory diagnostic
- `continual_qa`
- methods: `standard_rag`, `gap_current`, `gap_memory_keyed`, `gap_memory_ema`

## 6. 해석 규칙

- `FEVER` positive + `NQ/HotpotQA` negative
  - discrepancy control is verification-oriented, not generation-oriented

- `continual_qa` negative
  - persistent memory is not currently a supported core claim
