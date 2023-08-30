---
title: "[Hugging Face] Transformers 모델 아키텍처"
header:
#   teaser: /assets/HF_pipeline/2022-08-28/%EC%9D%B4%EB%AF%B8%EC%A7%80_0827020.jpg
#   overlay_image: /assets/HF_pipeline/2022-08-28/%EC%9D%B4%EB%AF%B8%EC%A7%80_0827020.jpg
  overlay_filter: 0.5

categories:
  - NLP
tags:
  - [Hugging Face Transformers 모델 아키텍처]
comments: true
toc: true
toc_sticky: true
 
date: 2023-08-30
last_modified_at: 2023-08-30
---


# ****How do Transformers work?****

# 1. ****A bit of Transformer history****

![이미지 0830001.jpg](assets/HF_architecture/이미지 0830001.jpg)

- 트랜스포머 아키텍쳐(Attention is All You Need)는 17년 6월에 나옴
- 해당 아키텍처 연구의 초점은 machine translation이었음
- 해당 모델이 소개된 이후 우수한 모델들이 추가적으로 도입됨

## 1) GPT

- 2018년 6월
- 최초의 pretrained Transformer 모델.
- 다양한 NLP 작업에 대해 fine tuned되어 사용되었고 당시 많은 태스크에서 최고 성능을 달성함

## 2) BERT

- 2018년 10월
- 또 다른 대규모 pretrained model
- better summaries of sentences

## 2) GPT-2

- 2019년 2월
- 윤리적인 문제로 인해 즉시 공개되지 않음
- 기존 GPT보다 규모가 더 크고 성능이 향상된 GPT 버전

## 3) DistillBERT

- 2019년 10월
- 60% 속도 향상 & 메모리 소비 40% 축소 → BERT 성능의 97%를 유지
- distilled BERT 버전

## 4) BART & T5

- 2019년 10월
- original Transformer model과 동일한 아키텍처를 사용
- two large pretrained models
- original Transformer 아키텍처를 사용한 최초의 사전 학습 모델)

## 5) GPT-3

- 2020년 5월
- finetune 없이도 다양한 작업을 수행할 수 있는 GPT-2의 더 큰 버전
- zero-shot learning

이 외에도 다양한 pre trained 모델이 존재

## 사전 학습 모델 세가지 범주화

1) GPT-like 모델(**auto-regressive** Transformer 모델)

2) BERT-like 모델(**auto-encoding** Transformer 모델)

3) BART/T5-like 모델(sequence-to-sequence Transformer 모델)

→ 이후에 다룰 예정

# 2. ****Transformers are language models****

- Transformer 모델(GPT, BERT, BART, T5 등)은 *language model* 로 학습됨
- 해당 모델들이 self-supervised learning 방식으로 large amounts of raw text에 대해 학습됨
- self-supervised learning은 목적 함수(objectives)가 모델의 입력에서 자동으로 계산되는 학습 유형임
- 사람이 데이터에 레이블을 지정할 필요가 없음
- This type of model develops a statistical understanding of the language it has been trained on, but it’s not very useful for specific practical tasks.
- 이로인해 pretrained model은 *transfer learning를 거치게 됨*
- 모델은 given task에 대해 supervised learning을 통해 fine tuning됨 (using human-annotated labels)

**1) An example of a task: *causal language modeling*** 

- n개의 이전 단어를 후 다음 단어를 예측하는 경우, 여기서 출력할 예측값은 과거 와 현재 입력값에 의존하지만 미래 입력값에는 의존하지 않기 때문에 이것을 *causal language modeling* 이라고 함
    
    ![이미지 0830002.jpg](assets/HF_architecture/이미지 0830002.jpg)
    

**2) Another example: masked language modeling**

- 문장에서 masked word를 예측하는 *masked language modeling*
    
    ![이미지 0830003.jpg](assets/HF_architecture/이미지 0830003.jpg)
    

# 3. ****Transformers are big models****

![이미지 0830004.jpg](assets/HF_architecture/이미지 0830004.jpg)

- 몇 가지 모델(ex:DistilBERT)을 제외하고, 더 나은 성능을 달성하기 위한 일반적인 전략은 모델의 크기와 사전 훈련된 데이터의 양을 늘리는 것임
- pretrained model, 특히 큰 모델을 학습하려면 많은 양의 데이터가 필요함
- 이는 시간과 컴퓨팅 리소스 면에서 비용이 많이 듦
    
    ![이미지 0830005.jpg](assets/HF_architecture/이미지 0830005.jpg)
    
- 대규모 모델 학습은 사진에서 보듯 environmental impact도 야기함
- 대규모의 사전 학습을 실행할 때 배출되는 이산화탄소의 양을 보여줌
- 최적의 하이퍼 파라미터(hyperparameter)를 얻기 위해 많은 학습 시도를 실행해야 하는 경우 탄소 발자국(carbon footprint)은 훨씬 더 높음
- 모든 연구자, 학생, 회사가 자신들의 모델을 직접 처음부터 사전 학습한다면 광범위하고 불필요한 비용이 생김
- 이러한 이유로 언어 모델을 공유해야만 함
- 학습된 가중치(weights)를 공유하고 이미 학습된 가중치를 기반으로 fine-tuning하여 모델을 만들면 커뮤니티의 전체 컴퓨팅 비용과 탄소 발자국(carbon footprint)을 줄일 수 있음

# 4. ****Transfer Learning****

## 1) Pretraining

- Pretraining은 모델을 처음부터 학습하는 작업임
- 모델의 가중치(weight)는 무작위로 초기화되고, 사전 지식(prior knowledge)이 없이 학습이 시작됨
    
    ![이미지 0830006.jpg](assets/HF_architecture/이미지 0830006.jpg)
    

- pretraining은 매우 많은 양의 데이터가 필요함
- 매우 큰 규모의 데이터 corpus가 필요하며 학습에는 최대 몇 주가 소요될 수 있음

## 2) fine-tuning

- 반면에 fine-tuning은 모델이 사전 학습된 **후에** 수행되는 학습임
- fine-tuning을 수행하려면 먼저 pretrained language model을 확보한 다음, 특정 태스크에 맞는 dataset을 사용하여 추가 학습을 수행함

### 최종 task를 위해 처음부터 직접 학습하지 않는 이유(= pre training의 장점)

1) pretrained model은 fine-tuning에 사용할 데이터셋과 유사한 데이터를 바탕으로 이미 학습됨

따라서, fine-tuning 과정에서, 사전 학습 과정에서 얻은 지식을 활용할 수 있음 (ex: NLP 문제의 경우 pretrained model은 원하는 task에 사용하는 언어에 대한 일종의 통계적 이해를 가지게 됨.)

2) 사전 학습된 모델은 이미 많은 데이터에 대해 학습되었기 때문에 fine-tuning 과정에서 훨씬 적은 데이터셋을 이용하더라도 좋은 결과를 얻을 수 있음

3) 좋은 결과를 얻는 데 필요한 시간과 자원은 훨씬 적을 수 있음

- 예를 들어, 영어로 사전 학습된 모델을 가지고, arXiv 말뭉치에서 fine-tuning하여 science/research 분야 데이터에 특화된 모델을 만들 수 있음
- fine-tuning에는 제한된 양의 데이터만 필요함
- 사전 학습된 모델이 획득한 지식은 transferred되므로 *전이 학습(transfer learning)* 이라는 용어를 사용함

![이미지 0830007.jpg](assets/HF_architecture/이미지 0830007.jpg)

- 따라서 모델을 fine-tuning하면 time, data, financial, environmental 비용이 절감됨
- 또한 학습 과정이 전체 pretraining보다 제약이 적기 때문에, 보다 쉽고 빠르게 다양한 미세 조정 작업을 반복할 수 있음
- 자신이 원하는 최종 task에 필요한 데이터가 충분치 않을 경우, fine-tuning 과정에서 처음부터 학습하는 방법보다 더 나은 결과를 얻을 수 있음. 이로 인해 pre trainied 모델을 활용하고 이를 fine tuning하는 과정이 필요함

# 5. ****General architecture****

- Transformer 모델의 일반적인 아키텍처를 살펴보고자 함.

## 1) **개요 (Introduction)**

![이미지 0830008.jpg](assets/HF_architecture/이미지 0830008.jpg)

- 모델은 두 개의 블록으로 구성됨

**1) 인코더(Encoder)**: 

- The encoder receives an input and builds a **representation** of it **(its features).**
- This means that the model is **optimized to acquire understanding** from the input.
- input에 대한 understanding에 대해 최적화

**2) 디코더(Decoder)**:

- The decoder uses the **encoder’s representation (features) along with other inputs** to generate a target sequence.
- This means that the model is optimized for generating outputs.
- 출력 생성에 최적화

→ 각각의 블록은 작업의 종류에 따라 개별적으로 사용할 수도 있음

1) **인코더 전용 모델(Encoder-only models)**: 

- sentence classification & named-entity recognition과 같이 입력에 대한 분석 및 이해(understanding)가 필요한 태스크에 적합

2) **디코더 전용 모델(Decoder-only models)**: 

- text generation 과 같은 generative tasks에 좋음

3) **Encoder-decoder models** or **sequence-to-sequence models**:

- translation 이나 summarization과 같이 입력이 수반되는 generative tasks에 적합함

## 2) ****Attention layers****

- Transformer 모델의 가장 중요한 특징은 *어텐션 레이어(attention layers)* 라는 특수 레이어로 구축된다는 부분
- attention layer**가 각 단어의 표현을 처리할 때, 문장의 특정 단어들에 특별한 주의를 기울이고 나머지는 거의 무시하도록 모델에 지시함**

**<예시: 영어 → 불어>**

- “You like this course"라는 입력이 주어지면 번역 모델은 "like"라는 단어에 대한 적절한 번역을 얻기 위해 인접 단어 "You"에도 주의를 기울여야 함
- 왜냐하면 프랑스어에서 동사 "like"는 주어(subject)에 따라 다르게 활용되기 때문임
- 그러나 문장의 나머지 부분("this course")은 해당 단어("like")의 번역에 그다지 유용하지 않음
- 같은 맥락에서, "this"를 번역할 때, 모델은 "course"라는 단어에도 주의를 기울여야 함
- "this"는 연결된 명사가 남성(masculine)인지 여성(feminine)인지에 따라 다르게 번역되기 때문입니다. 위와 마찬가지로, 문장의 다른 단어들("You", "like")은 "this"의 번역에 중요하지 않습니다. 더 복잡한 문장이나 더 복잡한 문법 규칙의 경우, 모델은 개별 단어를 적절하게 번역하기 위해 문장에서 해당 단어와 멀리 떨어진 단어에도 특별한 주의를 기울여야 함

→ 동일한 개념이 자연어와 관련된 모든 태스크에 적용됨

단어 자체가 고유한 의미를 가지고 있지만 그 의미는 주변 문맥 (context)에 의해 크게 영향을 받으며, 컨텍스트는 처리 중인 단어 앞이나 뒤에 존재하는 다른 단어들을 포함할 수 있음

## 3) ****The original architecture****

- Transformer 아키텍처는 번역용으로 설계됨
- During training, **the encoder receives inputs (sentences) in a certain language**, 
while the **decoder receives the same sentences** in the **desired target language.**
- 학습이 진행되는 동안 encoder는 특정 언어로 표기된 입력(문장)을 받고, 
디코더(decoder)는 원하는 대상 언어로 표기된 동일한 의미의 문장을 받음
- **In the encoder**, the attention layers can use **all the words** in a sentence (since, as we just saw, the translation of a given word can be dependent on what is after as well as before it in the sentence).
- 인코더에서 어텐션 레이어(attention layer)는 문장의 모든 단어에 주의(attention)를 기울일 수 있음. 왜냐하면, 현재 단어에 대한 번역 결과는 문장에서 해당 단어의 앞부분과 뒷부분의 내용에 따라 달라질 수 있기 때문임
- **The decoder**, however, works sequentially and can **only pay attention to the words** in the sentence that it has already translated (so, only the words before the word currently being generated). 
- For example, when we have predicted the first three words of the translated target, we give them to the decoder which then uses all the inputs of the encoder to try to predict the fourth word.
- 그러나 디코더는 순차적으로 작동하며 이미 번역된 문장의 단어들에만, 즉 현재 **생성되고 있는 단어 앞의 단어들에만 주의(attention)**를 기울일 수 있습니다.
예를 들어, 번역 대상(target sentence)의 처음 세 단어를 예측한 경우, 디코더에 이를 입력한 다음 인코더의 모든 입력(원본 문장의 모든 단어들)을 사용하여 네 번째 단어를 예측하려고 시도함
- To speed things up during training (when the model has access to target sentences), the decoder is fed the whole target, but it is not allowed to use future words (if it had access to the word at position 2 when trying to predict the word at position 2, the problem would not be very hard!). For instance, when trying to predict the fourth word, the attention layer will only have access to the words in positions 1 to 3.
- 학습 도중 속도를 높이기 위해, 디코더(decoder)는 전체 대상 문장(target sentences)을 입력으로 받지만, 이 중에서 미래 단어(현재 디코딩 대상 단어의 이후에 나타나는 단어들)를 사용하는 것은 허용되지 않음
- 두 번째 위치에 나타나는 단어를 예측하려고 할 때, 두 번째 위치의 정답 단어를 바로 접근할 수 있다면 학습이 제대로 진행되지 않을 것임. 예를 들어, 네 번째 단어를 예측하려고 할 때 어텐션 계층은 첫 번째에서 세번째 까지의 단어들에만 주의를 집중할 수 있습니다.

![이미지 0830010.jpg](assets/HF_architecture/이미지 0830010.jpg)

- Transformer 아키텍처: 왼쪽에 인코더, 오른쪽에 디코더
- the **first attention layer** in a decoder block pays attention to **all (past) inputs** to the decoder, but **the second attention layer** uses the **output of the encoder.** It can thus access the whole input sentence to best predict the current word.
- 디코더(decoder) 블록의 첫 번째 어텐션 계층(attention layer)은 디코더(decoder)에 대한 모든 과거 입력에 주의를 집중하지만, 두 번째 어텐션 계층은 인코더의 출력을 입력으로 받아서 사용함. 따라서 현재 단어를 가장 잘 예측하기 위해 전체 입력 문장(input/source sentence)에 액세스할 수 있음
- This is very useful as different languages can have grammatical rules that put the words in different orders, or some context provided later in the sentence may be helpful to determine the best translation of a given word.
- 이는 대상 언어(target language)가 원 언어(source language)와 비교하여 상당히 다른 단어 순서(words in different orders)로 문장을 표현하는 문법 규칙(grammatical rule)을 가지거나, 원본 문장(input/source sentence)의 뒷부분에 나타난 컨텍스트(context)가 현재 단어에 대한 최상의 번역을 결정하는 데 도움이 될 수 있는 경우 매우 유용함
- The *attention mask* can also be used in the encoder/decoder to prevent the model from paying attention to some special words — for instance, the special padding word used to make all the inputs the same length when batching together sentences.
- *어텐션 마스크(Attention mask)* 는 인코더/디코더에서 모델이 특정 단어에 주의를 집중하는 것을 방지하는 데 사용할 수 있음. 예를 들어, 문장을 일괄 처리(batching)할 때 모든 입력을 동일한 길이로 만들기 위해서 사용되는 특수 패딩 단어(padding word)에 적용할 수 있음

## 4) ****Architectures vs. checkpoints****

*아키텍처(architecture), 체크포인트(checkpoint)*, 모*델(model)* 용어

**1) 아키텍처(Architectures):** 

- 모델의 뼈대(skeleton)를 의미
- 모델 내에서 발생하는 각 레이어(layer)와 오퍼레이션(operation, 연산) 등을 정의함

**2) 체크포인트(Checkpoints):** 

- 해당 아키텍처에서 로드될 가중치 값

**3) 모델(Model):** 

- 포괄적인 용어(umbrella term)로 아키텍처나 체크포인트 두 가지 모두를 의미할 수도 있음
- 강좌에서는 표기의 명확성이 필요할 경우 모델이라는 용어보다는 아키텍처(architecture) 또는 체크포인트(checkpoint)를 주로 사용할 예정

→ 예를 들어, BERT는 아키텍처(architecture)이고 BERT의 첫 번째 릴리스를 위해 Google 팀에서 학습한 가중치 세트(set of weights)인 **bert-base-cased** 는 체크포인트(checkpoint)임. 하지만, "BERT 모델(BERT model)"과 "**bert-base-cased**"도 모델이라고 말할 수도 있음