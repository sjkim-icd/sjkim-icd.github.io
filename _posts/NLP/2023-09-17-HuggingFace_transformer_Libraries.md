---
title: "[Hugging Face] 2장 소개 및 Transformers 라이브러리 특징"
header:
  overlay_filter: 0.5

categories:
  - NLP
tags:
  - [Hugging Face 2장 소개 및 Transformers 라이브러리 특징]
comments: true
toc: true
toc_sticky: true
 
date: 2023-09-17
last_modified_at:  2023-09-17
---

# [Hugging Face][C-2] 2장 소개 및 Transformers 라이브러리 특징

# 라이브러리 사용하기

- Transformer 모델은 일반적으로 규모가 매우 큼
- 수백만에서 수천억 개의 매개변수가 포함된 모델을 학습하고 배포하는 일은 매우 복잡한 작업
- 새로운 거의 매일 출시되고 각각 고유한 구현 방식이 있기 때문에, 이 모든 모델들을 시험해 보는 것 또한 쉬운 일이 아님
- 이를 해결하기 위해 Transformers 라이브러리가 만들어짐 
→ 모든 Transformer 모델들을 적재, 학습하고, 저장할 수 있는 단일 API를 제공함

## Transformers 라이브러리의 특징

### 1) **사용 용이성(Ease of use)**:

- 최신 NLP 모델을 기반으로 추론 작업을 수행하기 위해서, 해당 모델을 다운로드, 적재 및 사용하는데 단 두 줄의 코드만 작성하면 됨

### 2) **유연성(Flexibility)**:

- PyTorch의 `nn.Module` 또는 TensorFlow의 `tf.keras.Model` 클래스로 표현됨
- 각 기계 학습(ML) 프레임워크(framework, e.g., PyTorch, Tensorflow) 내에서의 다른 모델들과 동일하게 취급됨

### 3) **단순성(Simplicity)**:

- 라이브러리 전체에서 추상화(abstraction)가 거의 이루어지지 않음
- "All in one file"은 Transformers 라이브러리의 핵심 개념
- 다시 말해서, 모델의 forward pass가 단일 파일에 완전히 정의되어 해당 코드 자체를 쉽게 이해할 수 있음
- 이 마지막 특징은 Transformers 라이브러리가 다른 기계학습 라이브러리와 구별되는 차별성임
- 모든 모델은 파일 간에 공유되는 모듈에 기반하여 구현되지 않음
- 대신 각 모델에는 자체 레이어가 있음. 모델을 더 접근하기 쉽고 이해하기 쉽게 만드는 것 외에도, 다른 모델에 영향을 주지 않고 특정 모델에서 쉽게 실험을 진행할 수 있음

# 2장 목표

- 2장에서는 1장에서 소개한 `pipeline()` 함수를 대체하기 위해, 직접 모델(model)과 토크나이저(tokenizer)를 함께 사용하는 end-to-end 예제를 활용함

**1) 모델 API**

- 모델 클래스 및 configuration 클래스를 자세히 살펴보고, 모델을 로드하는 방법과 모델이 예측(prediction)을 출력하기 위해 numerical input을 처리하는 방법을 알아보고자 함

**2) tokenizer API** 

- 그런 다음, `pipeline()` 함수의 또 다른 주요 구성 요소인 토크나이저(tokenizer) API를 살펴보고자 함
- 토크나이저는 신경망 모델의 입력으로 사용하기 위해서 텍스트 입력을 수치 데이터(numerical data)로 변환하고, 필요시 변환된 수치 데이터를 다시 텍스트로 변환하는 기능을 수행할 수 있음
- 전체 프로세스의 처음과 마지막을 담당함
- 마지막으로, 배치(batch) 형태로 여러 문장들을 모델로 한꺼번에 입력하는 방법을 알아보고, 상위 수준의 `tokenizer()` 함수를 자세히 살펴보고 자함