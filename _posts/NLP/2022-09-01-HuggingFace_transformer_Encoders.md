---
title: "[Hugging Face][C-1] Encoder models"
header:
#   teaser: //assets/HF_pipeline/2022-08-28/%EC%9D%B4%EB%AF%B8%EC%A7%80_0827020.jpg
#   overlay_image: //assets/HF_pipeline/2022-08-28/%EC%9D%B4%EB%AF%B8%EC%A7%80_0827020.jpg
  overlay_filter: 0.5

categories:
  - NLP
tags:
  - [Hugging Face Transformers Encoder models]
comments: true
toc: true
toc_sticky: true
 
date: 2023-09-01
last_modified_at: 2023-09-01
---



# 1. Encoder Models

- encoder models은 Transformers 모델의 인코더 모듈만 사용함
- 각 단계에서 attention layer은 initial sentence의 모든 단어에 접근할 수 있음, 이는 bi-directional attention을 수행하고 *auto-encoding model* 이라고 부르기도 함
- 모델의  pretraining 과정에서 주어진 초기 문장을 다양한 방법을 사용하여 corrupting하고(예: 임의의 단어를 masking), 손상시킨 문장을 다시 원래 문장으로 복원하는 과정을 통해서 모델 학습함
- encoder models은 sentence classification, named-entity recognition, word classificationextractive question answering 등과 같이 전체 문장에 대한 이해가 필요한 작업(task)에 가장 적합함

## Encoder models 종류

- **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)**
- **[BERT](https://huggingface.co/transformers/model_doc/bert.html)**
- **[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)**
- **[ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)**
- **[RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)**

# 2. Transformer models Encoder with Lysandre

![이미지 0831016.jpg](/assets/HF_C1/이미지 0831016.jpg)

- 인코더 전용 아키텍처의 예로는 BERT가 있음
    
    ![이미지 0831017.jpg](/assets/HF_C1/이미지 0831017.jpg)
    
- 작동 방식을 이해하는 것부터 시작하고자 함
    
    ![이미지 0831018.jpg](/assets/HF_C1/이미지 0831018.jpg)
    
- 예제로 3가지 단어를 입력으로 사용하고 인코더를 통해 전달함
- 각 단어의 숫자 표현을 검색함
- 예를 들어 ‘Welcome to NYC’라는 세 단어를 변환함
    
    ![이미지 0831019.jpg](/assets/HF_C1/이미지 0831019.jpg)
    
- 인코더는 입력 단어당 정확히 하나의 숫자 시퀀스를 출력함
- 이 숫자 표현은 feature vector or feature tensor라고 함
    
    ![이미지 0831020.jpg](/assets/HF_C1/이미지 0831020.jpg)
    
- 이 feature에 대해 살펴보면 여기에 인코더를 통해 전달된 단어당 하나의 벡터가 포함됨
- 이러한 각 벡터는 단어를 숫자로 표현한 것임
- 해당 벡터의 차원은 모델의 아키텍처에 의해 정의됨
- 기본 BERT 모델의 경우, 768임
- 이러한 representaion에는 단어의 값이 포함됨. contextualize되어있음
- 예를 들어 ‘to’라는 단어에 귀속된 벡터는 ‘to’ 단어만을 표현한 것이 아님
- context라고 불리는 주변 단어도 고려함
- left context → 타겟 단어 왼쪽에 있는 단어 → 여기서는 welcome이라는 단어
- right context → 타겟 단어 오른쪽에 있는단어 → 여기서는 NYC라는 단어
- 를 입력하고 해당 context 내에서 해당 단어의 값을 출력함
- 그래서 contextualize가 되어있음 768개 값의 벡터가 텍스트에 있는 해당 단어의 의미를 담고 있다고 말할 수 있음
- 이를 수행할 수 있는 건 self attention 매커니즘 덕분임
- self attention 매커니즘은 해당 시퀀스의 표현을 계산하기 위해 단일 시퀀스의 다양한 위치(다양한 단어)와 관련됨
- 이는 단어의 representation이 시퀀스의 다른 단어에 의해 영향을 받았다는 것을 의미함

## 언제 인코더를 사용해야할까?

![이미지 0901001.jpg](/assets/HF_C1/이미지 0901001.jpg)

- 인코더는 다양한 작업에서 독립적으로 사용할 수 있음
- 가장 유명한 변환기 모델인 BERT는 독립형 인코더 모델이며, 나올 당시에 질문답변 작업, 시퀀스 분류 작업 , masked된 언어 모델링에서 성능이 좋았음
- 인코더는 시퀀스에 대한 의미있는 정보를 전달하는 벡터를 추출하는데 매우 강력한 아이디어임
- 추가적인 뉴런 레이어도 추가할 수 있음

### Maksed Language Modeling(MLM)

![이미지 0901002.jpg](/assets/HF_C1/이미지 0901002.jpg)

- 일련의 단어에서 숨겨진 단어를 예측하는 작업임
- My와 is 사이의 단어를 mask함
- 숨겨진 단어를 순차적으로 예측하도록 학습됨
- 특히 이 시나리오에서는 인코더가 빛을 발하는데, 이 task의 경우, 양방향 정보가 중요하기 떄문임
- 오른쪽 단어 is, Syvain, .이 없다면 BERT가 name을 올바른 단어로 식별할 가능성은 거의 없음
- 인코더는 마스킹된 단어를 예측하기 위해 시퀀스를 잘 이해해야함
- 왜냐면 텍스트가 문법적으로 정확하더라도 시퀀스의 맥락에서 반드시 의미가 있는 것은 아니기 때문임

### Sentiment analysis

![이미지 0901003.jpg](/assets/HF_C1/이미지 0901003.jpg)

- 인코더는 시퀀스 분류에 능숙함
- 감성 분석은 순서 분류 작업의 예시임
- 모델의 목표는 시퀀스의 감정을 식별하는 것임
- 리뷰 분석을 진행하는 경우, 시퀀스에 별 1~5개 등급을 부여하는 것부터 여기에 표시된 대로 시퀀스에 긍/부정 등급을 부여하는것 까지 범위가 다양함
- 여기에 두 개의 시퀀스가 있는 경우, 모델을 사용하여 예측을 계산하고 양,음성의 2 클래스 중 시퀀스를 분류함
- 두 시퀀스는 동일한 단어를 포함하여 매우 유사하지만 의미는 다르며 인코더 모델은 그 차이를 파악할 수 있음

# References

[https://www.youtube.com/watch?time_continue=3&v=MUqNwgPjJvQ&embeds_referring_euri=https%3A%2F%2Fhuggingface.co%2F&feature=emb_logo](https://www.youtube.com/watch?time_continue=3&v=MUqNwgPjJvQ&embeds_referring_euri=https%3A%2F%2Fhuggingface.co%2F&feature=emb_logo)