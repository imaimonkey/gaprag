# GapVerify Code Restructure Plan

## 1. 목적

현재 코드베이스는 여전히 `persistent continual RAG` 서사가 강하게 남아 있습니다.
이 문서는 verification 중심 프로젝트로 전환하기 위한 구조 재편 방향을 정리합니다.

## 2. 유지할 핵심 모듈

### 유지
- `gapverify/retriever.py`
- `gapverify/generator.py`
- `gapverify/hidden_extractor.py`
- `gapverify/doc_encoder.py`
- `gapverify/gap_estimator.py`
- `gapverify/gap_injector.py`
- `gapverify/pipeline.py`
- `scripts/run_eval.py`
- `scripts/build_index.py`

이 모듈들은 current-gap verification 파이프라인의 중심입니다.

## 3. 보조/진단 모듈로 내릴 것

### 축소
- `gapverify/gap_memory.py`
- `scripts/run_continual_eval.py`
- `configs/continual_qa.yaml`
- `memory` 관련 ablation 전체

이 부분들은 삭제 대상은 아니지만, 메인 contribution 경로에서는 내려야 합니다.

## 4. 다음 구조 개편 우선순위

### Phase 1: 문서/실험 관점 정렬
- README와 manual을 verification 중심으로 정렬
- FEVER를 main benchmark로 승격
- `continual_qa`를 diagnostic benchmark로 명시

### Phase 2: benchmark adapter 확장
- `AVeriTeC` dataset adapter 추가
- `HoVer` dataset adapter 추가
- `FEVEROUS` dataset adapter 추가

### Phase 3: verification 전용 분석 강화
- label calibration
- support/refute/conflict breakdown
- retrieval hit vs verdict correctness
- changed output vs changed verdict 분리

### Phase 4: current-gap 전용 method 강화
- verification-specific injection variants
- verdict-focused confidence analysis
- per-label gap distribution logging

## 5. 파일 단위 수정 후보

### 우선 수정
- `scripts/prepare_benchmark_data.py`
  - verification benchmark adapters 확장
- `scripts/run_experiments.sh`
  - verification suite preset 추가
- `scripts/analyze_results.py`
  - verification 중심 plot/table 추가
- `configs/`
  - FEVER/AVeriTeC/HoVer/FEVEROUS 전용 config 추가

### 이후 정리 가능
- `rag.yaml`, `gapverify_memory.yaml`, `gapverify_current.yaml` 명명 재정리
- `continual_qa` 관련 naming을 appendix/diagnostic 성격으로 명확화

## 6. 추천 구조 목표

### 메인 실험선
- `standard_rag`
- `gap_current`
- benchmarks: `FEVER`, `AVeriTeC`, `HoVer`, `FEVEROUS`

### 보조 실험선
- `gap_memory_keyed`
- `gap_memory_ema`
- benchmark: `continual_qa`

### 경계 분석선
- `NQ`
- `HotpotQA`

## 7. 최종 상태 목표

최종적으로 이 프로젝트는
- verification-oriented mainline
- QA negative-transfer analysis
- memory diagnostic appendix

의 3개 축으로 읽히는 구조가 되어야 합니다.
