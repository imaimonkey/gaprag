# GapVerify Research Scope

## 1. 연구 정의

GapVerify는 **retrieval-grounded verification에서 latent discrepancy를 training-free control signal로 사용하는 방법**을 연구합니다.

핵심 객체는 세 가지입니다.
- retrieved evidence
- model-internal belief state
- explicit latent discrepancy between them

이 프로젝트의 기본 가정은 다음과 같습니다.
- evidence를 본 모델 상태와 query/claim 중심 상태 사이에는 의미 있는 discrepancy가 존재할 수 있다.
- 이 discrepancy는 inference-time에서 직접 활용 가능한 control signal일 수 있다.
- 이 신호의 효과는 태스크와 evidence 구조에 따라 달라질 수 있다.

## 2. 핵심 연구 물음

1. retrieved evidence와 모델 내부 state 사이의 latent discrepancy를 안정적으로 추출할 수 있는가?
2. 이 discrepancy를 주입하면 verification verdict 형성에 도움이 되는가?
3. 이 효과는 어떤 benchmark family에서 가장 잘 나타나는가?
4. stronger retrieval/generation config로 갈수록 이 signal은 유지되는가?

## 3. 핵심 가설

### 가설 A
retrieved evidence는 support, refute, uncertainty와 관련된 latent discrepancy를 유도한다.

### 가설 B
이 discrepancy는 verdict formation을 조정하는 training-free signal로 사용할 수 있다.

### 가설 C
signal의 품질은 benchmark의 evidence 구조, retrieval difficulty, verdict space에 따라 달라진다.

## 4. 방법론 축

### 메인 비교
- `standard_rag`
- `gap_current`

### secondary branch
- `gap_memory_keyed`
- `gap_memory_ema`

이 문서 기준에서 primary method는 `gap_current`입니다.

## 5. 벤치마크 구조

### verification core
- `FEVER`
- `HoVer`
- `FEVEROUS`
- `AVeriTeC`

### transfer boundary
- `NQ`
- `HotpotQA`

### memory diagnostic
- `continual_qa`

## 6. 현재 실험 프레임

프로젝트의 기본 프레임은 다음과 같습니다.
1. verification benchmark에서 baseline 대비 gain 측정
2. stronger preset에서 robustness 확인
3. QA benchmark에서 transfer boundary 확인
4. memory branch에서 behavior diagnostic 확인

## 7. 주장 범위

현재 이 프로젝트에서 직접 겨냥하는 주장은 다음입니다.
- latent discrepancy can act as a training-free control signal for retrieval-grounded verification
- the effect should be evaluated under multiple benchmark families and stronger config variants

즉 이 문서는 과거 경로 설명이 아니라, 현재 `GapVerify`가 어떤 연구 질문을 수행하는 코드베이스인지 정의하는 문서입니다.
