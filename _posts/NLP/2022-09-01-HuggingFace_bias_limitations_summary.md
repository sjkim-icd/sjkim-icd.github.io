---
title: "[Hugging Face][C-1]  Bias and limitations  & C-1 Summary"
header:
#   teaser: //assets/HF_pipeline/2022-08-28/%EC%9D%B4%EB%AF%B8%EC%A7%80_0827020.jpg
#   overlay_image: //assets/HF_pipeline/2022-08-28/%EC%9D%B4%EB%AF%B8%EC%A7%80_0827020.jpg
  overlay_filter: 0.5

categories:
  - NLP
tags:
  - [Hugging Face Bias and limitations  & C-1 Summary]
comments: true
toc: true
toc_sticky: true
 
date: 2023-09-01
last_modified_at: 2023-09-01
---


# 1. B**ias and limitations**

- 사전 학습된 모델이나 미세 조정된 모델을 상용 시스템에서 사용하려는 경우, 제약 사항이 존재함
- 대용량 데이터에 대해서 사전 학습을 수행하기 위해, 인터넷에 존재하는 좋은 데이터는 물론 최악의 데이터들도 무조건 수집하여 활용해야 함
- 예제) BERT 모델을 활용한 `fill-mask` 파이프라인의 예시

```python
from transformersimport pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"]for rin result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"]for rin result])

```

```
['lawyer', 'carpenter', 'doctor', 'waiter', 'mechanic']
['nurse', 'waitress', 'teacher', 'maid', 'prostitute']
```

- 위 두 문장에서 누락된 단어를 채우라는 요청을 받았을 때 모델은 성별과 상관없는(gender-free) 대답(waiter/waitress)은 오직 하나만 제공함.
- 나머지는 일반적으로 특정 성별과 밀접한 관련이 있는 직업.
- 매춘부(prostitute)는 모델이 "여성" 및 "직업"과 연관되는 상위 5개 단어에 속해 있음.
- 이러한 현상은 BERT가 인터넷 전체에서 데이터를 수집하여 학습된 것이 아니라, 오히려 중립적인 데이터 즉, [English Wikipedia](https://huggingface.co/datasets/wikipedia) 와 [BookCorpus](https://huggingface.co/datasets/bookcorpus)를 사용하여 학습된 드문 Transformer 모델 중 하나임에도 불구하고 발생함
- When you use these tools, you therefore need to keep in the back of your mind that the original model you are using could very easily generate sexist, racist, or homophobic content. Fine-tuning the model on your data won’t make this intrinsic bias disappear.

# 2. Course1 Summary

- high-level `pipeline()` 함수를 사용하여 다양한 NLP 작업을 수행하는 방법을 살펴보았음
- 허브에서 모델을 검색하고 사용하는 방법과 Inference API를 사용하여 브라우저에서 직접 모델을 테스트하는 방법을 알아봄
- 또한 Transformer 모델이 high-level에서 어떻게 작동하는지 논의하고 전이 학습(transfer learning)과 미세 조정(fine-tuning)의 중요성에 대해 이야기함
- 가장 중요한 핵심은 대상 작업의 종류에 따라 전체 아키텍처(full architecture)를 사용하거나 인코더(encoder) 또는 디코더(decoder)만 사용할 수 있다는 부분임
- 위 내용을 요약한 표
    
    
    | Model | Examples | Tasks |
    | --- | --- | --- |
    | Encoder | ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa | Sentence classification, named entity recognition, extractive question answering |
    | Decoder | CTRL, GPT, GPT-2, Transformer XL | Text generation |
    | Encoder-decoder | BART, T5, Marian, mBART | Summarization, translation, generative question answering |