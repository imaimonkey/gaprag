# GapVerify Research Redefinition

## 1. 왜 재정의가 필요한가

초기 프로젝트 정의는 다음을 전제로 했습니다.
- persistent latent discrepancy memory
- training-free continual RAG
- future query improvement through memory accumulation

하지만 현재 구현과 결과는 이 강한 서사를 지지하지 않습니다.

실제 결과 요약:
- `gap_current`는 `FEVER`에서 긍정 신호를 보임
- `gap_current`는 `NQ`, `HotpotQA`에서 악화
- `gap_memory_keyed`, `gap_memory_ema`는 `continual_qa`에서 gain을 재현하지 못함

따라서 메인 연구 질문을 바꿔야 합니다.

## 2. 새로운 연구 정의

**GapVerify is a research project on latent discrepancy control for retrieval-grounded verification.**

즉 핵심은:
- `persistent memory`가 아니라 `latent discrepancy`
- `continual QA`가 아니라 `verification`
- `future-query improvement`가 아니라 `current-query control`

## 3. 새로운 연구 물음

1. retrieved evidence와 모델 내부 belief state 사이의 latent discrepancy를 안정적으로 추출할 수 있는가?
2. 이 discrepancy를 training-free inference-time control signal로 사용하면 verification 정확도가 개선되는가?
3. 왜 이 신호는 free-form QA보다 label-style verification에서 더 잘 작동하는가?

## 4. 핵심 가설

### 가설 A
retrieved evidence는 support/refute/uncertain과 관련된 latent discrepancy를 유도한다.

### 가설 B
이 discrepancy는 긴 자연어 answer generation을 안정화하는 신호는 아니지만, evidence-grounded decision boundary를 움직이는 신호일 수 있다.

### 가설 C
따라서 discrepancy injection은 general QA enhancement보다 verification control 쪽에 더 적합하다.

## 5. 메인 방법론

### 중심 방법
- `standard_rag`
- `gap_current`

### 보조 진단
- `gap_memory_keyed`
- `gap_memory_ema`

현재 단계에서 memory 계열은 핵심 기여가 아니라 negative-result/diagnostic branch로 취급한다.

## 6. 벤치마크 역할 재정의

### 메인 벤치
- `FEVER`
- `AVeriTeC`
- `HoVer`
- `FEVEROUS`

### 경계 분석 벤치
- `NQ`
- `HotpotQA`

### 진단 벤치
- `continual_qa`

## 7. 주장 범위

현재 허용되는 주장:
- latent discrepancy can be used as a training-free control signal for verification-style tasks
- the same signal does not cleanly transfer to free-form QA

현재 허용되지 않는 주장:
- persistent memory robustly improves continual RAG
- gap injection is a general-purpose QA improvement method
- current benchmark coverage is sufficient for a final verification paper

## 8. 논문/보고서 서사

추천되는 서사 구조:
1. discrepancy extraction
2. verification gain on FEVER-like tasks
3. negative transfer on free-form QA
4. memory branch as unresolved or negative finding

## 9. 실용적 의미

이 재정의는 프로젝트를 폐기하는 것이 아니라, 결과와 맞지 않는 강한 가설을 버리고 결과와 정합적인 질문으로 이동하는 것이다.
