---
title: "[Hugging Face][C-1] Decoder models"
header:
#   teaser: //assets/HF_pipeline/2022-08-28/%EC%9D%B4%EB%AF%B8%EC%A7%80_0827020.jpg
#   overlay_image: //assets/HF_pipeline/2022-08-28/%EC%9D%B4%EB%AF%B8%EC%A7%80_0827020.jpg
  overlay_filter: 0.5

categories:
  - NLP
tags:
  - [Hugging Face Transformers Decoder models]
comments: true
toc: true
toc_sticky: true
 
date: 2023-09-01
last_modified_at: 2023-09-01
---

# Decoder models

- decoder models은 Transformer 모델의 디코더 모듈만 사용함
- 각 단계에서 주어진 단어에 대해 attention layer는 문장에서 현재 처리 단어 앞쪽에 위치한 단어들에만 액세스할 수 있음. 이러한 모델을 *auto-regressive models이라고 함*
- decoder models의 pretraining은 일반적으로 문장의 다음 단어 예측을 수행함으로써 이루어짐.
- 이러한 모델은 text generation과 관련된 작업(task)에 가장 적합함

## Decoder 모델의 종류

- **[CTRL](https://huggingface.co/transformers/model_doc/ctrl.html)**
- **[GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt)**
- **[GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)**
- **[Transformer XL](https://huggingface.co/transformers/model_doc/transfo-xl.html)**

# Transformer models: Decoder with Syvain

![이미지 0901004.jpg](/assets/HF_C1/이미지 0901004.jpg)

- 디코더 아키텍처에 대해 살펴보고자 함
- 널리 사용되는 디코더 전용 아키텍처의 예는 GPT-2가 있음

## Decoder 작동방식

![이미지 0901005.jpg](/assets/HF_C1/이미지 0901005.jpg)

- 인코더와 디코더는 매우 유사함, 약간의 성능 하락이 있지만 인코더와 동일한 작업 대부분에 디코더를 사용할 수 있음
- 인코더와 디코더의 구조적 차이점에 대해 이해를 하기 위해 인코더에 대해 취했던 것과 동일한 접근 방식을 취하고자 함
- 3단어를 디코더를 통해 전달함
    
    ![이미지 0901006.jpg](/assets/HF_C1/이미지 0901006.jpg)
    
    ![이미지 0901007.jpg](/assets/HF_C1/이미지 0901007.jpg)
    
- 우리는 각 단어의 숫자 표현을 검색함
- 세 개의 숫자 시퀀스에서 Welcome to NYC라는 세 단어를 변환함
- 디코더는 입력 단어당 정확히 하나의 숫자 시퀀스를 출력함
- 이 numerical representation은 ‘feature vector’ 또는 feature tensor라고 함

![이미지 0901009.jpg](/assets/HF_C1/이미지 0901009.jpg)

- 여기에는 디코더를 통과한 단어당 하나의 벡터가 포함됨
- 이들 벡터 각각은 문제의 단어를 수치로 표현한 것임

![이미지 0901010.jpg](/assets/HF_C1/이미지 0901010.jpg)

- 해당 벡터의 차원은 모델 아키텍처에 의해 정의됨
- 디코더가 인코더와 다른 점은 self-attention 매커니즘임
- masked self attention을 사용함
- 예를 들어 to라는 단어에 초점을 맞추면 해당 벡터가 NYC 단어에 의해 전혀 수정되지 않음을 알 수 있는데 그 이유는 단어의 right에 있는 모든 단어가 mask 되어있기 때문임
- 디코더는 왼쪽과 오른쪽에 있는 모든 단어, 즉 양방향 context의 이점을 누리는 대신에 왼쪽에 단어에만 액세스 가능함
- masked self attention은 추가 mask를 사용하여 단어 양쪽의 context를 숨긴다는 점에서 self attention 매커니즘과 다름
- 단어의 숫자 표현은 mask된 context의 단어에 영향을 받지 않음

## 디코더는 언제 사용해야할까?

![이미지 0901011.jpg](/assets/HF_C1/이미지 0901011.jpg)

- 인코더와 마찬가지로 디코더도 독립적으로 모델을 사용할 수 있음
- numerical representation을 생성하므로 다양한 작업에서 사용할 수 있음
- 디코더의 강점은 단어가 왼쪽 문맥에 접근하는 방식에 있음
- 왼쪽 컨텍스트에만 액세스할 수 있는 디코더는 텍스트 생성에 좋은 성능을 보임
- 즉, 알려진 단어 시퀀스가 주어지면 단어 또는 단어 시퀀스를 생성하는 기능임
- NLP에서는 이를 인과 언어 모델링(casual language modeling)

### casual language modeling 예제

![이미지 0901012.jpg](/assets/HF_C1/이미지 0901012.jpg)

- 모델링이 작동하는 방식의 예는 다음과 같음
- My라는 초기 단어로 시작함
- 디코더의 입력으로 사용함
- 모델은 차원 768의 벡터를 출력함
- 이 벡터에는 단일 단어 또는 단어인 시퀀스에 대한 정보가 포함되어 있음
- 해당 벡터에 작은 변환을 적용하여 모델에 의해 알려진 모든 단어에 매핑됨
- 나중에 볼 매핑 언어 모델링 헤드라고 함
- 우리는 모델이 가장 가능성이 높은 다음 단어가 name이라고 믿고 있음을 식별함
    
    ![이미지 0901013.jpg](/assets/HF_C1/이미지 0901013.jpg)
    
- 그런 다음 새 단어를 가져와 초기 시퀀스에 추가함
- My에서 이제 My name으로 이동함
- 여기서 auto regressive 측면이 등장함
- auto regressive model은 다음 단계에서 과거 출력을 입력으로 재사용함
- 다시 한번 똑같은 작업을 수행함
- 즉, 디코더를 통해 해당 시퀀스를 캐스팅하고 가장 가능성이 높은 다음 단어를 검색함
- 이제는 is라는 단어임

![이미지 0901015.jpg](/assets/HF_C1/이미지 0901015.jpg)

- 만족할 떄까지 작업을 반복함
- 단일 단어에서 시작하여 이제 full senetence를 생성함
- 더 진행할 수도 있는데 예를 들어 GPT-2의 최대 컨텍스트 크기는 1024이다
- 최대 1024개의 단어를 생성할 수 있음
- 그리고 디코더는 여전히 시퀀스의 첫번쨰 단어에 대한 일부 메모를 보유함