---
title: "[Hugging Face] NLP Task와 Transformers 라이브러리 pipeline 활용 사례"
header:
#   teaser: /assets/HF_pipeline/2022-08-28/%EC%9D%B4%EB%AF%B8%EC%A7%80_0827020.jpg
#   overlay_image: /assets/HF_pipeline/2022-08-28/%EC%9D%B4%EB%AF%B8%EC%A7%80_0827020.jpg
  overlay_filter: 0.5

categories:
  - NLP
tags:
  - [NLP Task와 Transformers 라이브러리 pipeline 활용 사례]
comments: true
toc: true
toc_sticky: true
 
date: 2023-08-29
last_modified_at: 2023-08-29
---
# [Hugging Face][C-1] NLP Task와 Transformers 라이브러리 pipeline 활용 사례

- Hugging Face의 Transformers, Datasets, Tokenizer, Accelerate 라이브러리, NLP에 대해 다루는 강의를 듣고 정리하고자 함
- Course1 트랜스포머 中 NLP와 **Transformers, what can they do?에 대한 내용을 정리함**

# 0. Welcome to the Hugging Face Course

![이미지 0829127.jpg](/assets/HF_C1/이미지 0829127.jpg)

- Hugging Face의 course는 다음과 같음
    
    ![이미지 0829128.jpg](/assets/HF_C1/이미지 0829128.jpg)
    
- Introduction, Diving in, Advanced로 나눠짐
- 1장에서 4장까지는 Transformers 라이브러리의 주요 개념 소개, Hugging Face Hub의 모델 사용 방법이나 데이터셋을 통한 fine tuning 및 Hub에 결과 공유하는 방법을 익히게 됨
- 5장에서 8장까지는 Datasets와 Tokenizers의 기초 숙지, 주요 NLP task를 다룸
- 9장에서 12장은 메모리 효율화 및 long sequences 문제 등 use case를 위한 custom objects를 사용하는 방법을 배움

![이미지 0829129.jpg](/assets/HF_C1/이미지 0829129.jpg)

- 챕터1 제외하고는 python과 DL 내용은 알고 있어야 함

# 1. **Natural Language Processing**

- NLP Task는 human language와 관련된 모든 것을 이해하는 데 중점을 둔 linguistics 및 machine learning의 한 분야
- NLP는 단일 단어를 개별적으로 이해하는 것 뿐만 아니라 해당 단어의 주변 문맥도 함께 이해할 수 있도록 하는 것을 목표로 함

## NLP Task의 종류

### 1) **Classifying whole sentences**

- 리뷰(review)의 감정(sentiment) 식별
- 스팸 이메일 감지
- 문장의 문법 & 문장 간에 논리적으로 관련되어 있는지 여부를 판단

### 2) **Classifying each word in a sentence**

- 문장의 문법적 구성요소(명사, 동사, 형용사)
- 명명된 개체(개체명, e.g., 사람, 위치, 조직) 식별

### 3) **Generating text content**

- 자동 생성된 텍스트로 프롬프트 완성(completing a prompt)
- 마스킹된 단어(masked words)로 텍스트의 공백 채우기

### 4) **Extracting an answer from a text**

- 질문(question)과 맥락(context)이 주어지면, 맥락에서 제공된 정보를 기반으로 질문에 대한 답변을 추출

### 5) **Generating a new sentence from an input text**

- 텍스트를 다른 언어로 번역(translation), 텍스트 요약(summarization)

→ NLP는 written text 뿐만 아니라 오디오 샘플의 스크립트(transcript) 또는 이미지 설명(image caption) 생성과 같은 음성 인식(speech recognition) 및 컴퓨터 비전(computer vision) 등의 복잡한 문제도 또한 해결함

→ "나는 배고프다(I am hungry)"와 "나는 슬프다(I am sad)"와 같은 두 문장이 주어지면 
인간은 두 문장이 얼마나 유사한지를 쉽게 결정할 수 있음
하지만 기계 학습의 경우 텍스트가 모델이 학습할 수 있는 방식으로 처리되어야 함

# 2. **Transformers, what can they do?**

- 트랜스포머 모델이 무엇을 할 수 있는지 살펴보고, Transformers 라이브러리의 첫 번째 도구인 pipeline을 사용하고자 함

## 1) Transformer와 Hugging Face

- 트랜스포머 모델은 모든 종류의 NLP 작업을 해결하는 데 사용됨
- Hugging Face 및 트랜스포머 모델을 사용하는 회사, 해당 회사들은 회사가 만든 모델들을 공유함
    
    ![이미지 0829130.jpg](/assets/HF_C1/이미지 0829130.jpg)
    

**(1) Hugging Face의 Transformers 라이브러리 GIT**

- 해당 라이브러리는 공유된 모델을 만들고 사용할 수 있는 기능을 제공함

[GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers)

**(2) Model Hub**

- 누구나 다운로드하여 사용할 수 있는 수천 개의 사전 학습된 모델(pretrained models) 존재
- 자신의 모델을 허브에 업로드도 가능함

[Models - Hugging Face](https://huggingface.co/models)

## 2) ****Working with pipelines****

- pipleine 함수는 특정 모델과 동작에 필요한 전처리 및 후처리 단계를 연결하여 
텍스트를 직접 입력하고 이해하기 쉬운 답변을 얻을 수 있음

### 예제 살펴보기 - sentiment-analysis

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

- pipeline 안에 task 내용과 문장을 입력함
    
    ![이미지 0829131.jpg](/assets/HF_C1/이미지 0829131.jpg)
    
- sentiment-analysis로 지정한 후에 문장을 넣었더니 POSITIVE하다고 판단 해줌
    
    ![이미지 0829132.jpg](/assets/HF_C1/이미지 0829132.jpg)
    
- 2개의 문장을 넣었을 때도 각각 어떤지 판단 해줌
    
    ![이미지 0829133.jpg](/assets/HF_C1/이미지 0829133.jpg)
    
- 한국어도 잘 판단해줌
- pipeline은 영어 문장에 sentiment analysis을 위해 fine-tuned pretrained model을 사용함
- `classifier` 객체를 생성할 때 모델이 다운로드됨, 생성된 `classifier` 객체를 다시 실행하면 다시 다운로드할 필요없이 캐시된 모델을 사용할 수 있음

### (1) 파이프라인의 **실행 3단계**

**1) preprocessing:** 

- 모델이 이해할 수 있는 형태로 텍스트는 전처리됨

**2) 전처리된 텍스트는 모델에 전달**

**3) postprocessing:**

- 모델이 예측한 결과는 postprocessing되어 우리가 이해할 수 있는 형태로 변환

### (2) 활용 가능한 파이프라인 종류

- `feature-extraction` (텍스트에 대한 벡터 표현)
- `fill-mask`
- `ner` (named entity recognition, 개체명 인식)
- `question-answering`
- `sentiment-analysis`
- `summarization`
- `text-generation`
- `translation`
- `zero-shot-classification`

이중 몇가지를 살펴보자

### (3) 파이프라인 종류 1- **Zero-shot classification**

- 레이블이 지정되지 않은 텍스트를 분류해야하는 Task
- 텍스트에 annotation을 추가하는 것은 time consuming & domain expertise 하기 때문에 zero shot은 많이 사용됨
- 해당 분류에 사용할 레이블을 직접 마음대로 지정할 수 있으므로 사전 훈련된 모델의 레이블 집합에 의존할 필요가 없음
- 두 레이블(긍정, 부정)을 사용하여 문장을 긍정 또는 부정으로 분류하는 걸 위에서 봄
- 사용자가 원하는 다른 레이블 집합을 사용하여 텍스트를 분류할 수도 있음

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

![이미지 0829134.jpg](/assets/HF_C1/이미지 0829134.jpg)

- 위 결과와 같이 해당 문장에 대해 label을 교육, 비즈니스, 정치로 하였는데 각각에 대해서 보여줄 수 있음
- 완전히 다른 새로운 레이블 집합으로 문장 분류를 수행할 때도 새로운 데이터를 이용해서 모델을 fine-tuning할 필요가 없기 때문에 *zero-shot* 분류라고 함
- 같이 원하는 레이블 목록에 대한 확률 점수 반환 가능

### (4) 파이프라인 종류 2- Text Generation

- 텍스트 생성 방법
- 입력으로 특정 prompt를 제공하면 모델이 나머지 텍스트를 생성하여 프롬프트를 자동 완성함

```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

![이미지 0829135.jpg](/assets/HF_C1/이미지 0829135.jpg)

- generator 객체에
- num_return_sequences→ 생성 시퀀스 갯수 지정
- max_length → 출력 텍스트 총 길이
    
    ![이미지 0829137.jpg](/assets/HF_C1/이미지 0829137.jpg)
    
- 파라미터를 각각 2,5로 지정해서 수행

### (5) Model Hub에 있는 모델을 지정하여 파이프라인 사용하기

- 이전까진 task에 대한 default model이 작동되었음
- Model Hub에 있는 모델을 지정할 수 있음
    
    ![이미지 0829138.jpg](/assets/HF_C1/이미지 0829138.jpg)
    
- Model Hub에 가서 원하는 task에 대한 tag를 선택하면 사용할 수 있는 모델이 표시됨(여기서는 Text Generation)
- 예제로 distilgpt2를 사용하자

```python
from transformers import pipeline

# distilgpt2 모델 로드
generator = pipeline("text-generation", model="distilgpt2")    
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

![이미지 0829139.jpg](/assets/HF_C1/이미지 0829139.jpg)

- distilgpt2를 사용한 결과

![이미지 0829140.jpg](/assets/HF_C1/이미지 0829140.jpg)

- Model Hub에서 language tags를 클릭하여 그 언어에 특화된 모델을 세부적으로 검색하고 선택함으로써 원하는 언어로 표현된 텍스트를 생성할 수 있는 모델을 사용할 수 있음
    - Model Hub에는 다중 언어를 지원하는 다국어 모델(multilingual models)에 대해서도 포함됨
    
    ![이미지 0829143.jpg](/assets/HF_C1/이미지 0829143.jpg)
    
- 한국어로 체크하고 download 최대를 보았더니 skt가 있어서 해당 모델 살펴봄
    
    ![이미지 0829145.jpg](/assets/HF_C1/이미지 0829145.jpg)
    
    - 특정 모델을 클릭하여 선택하면 온라인에서 직접 테스트할 수 있는 위젯(widget)이 표시됨 이렇게 하면 다운로드하기 전에 그 모델의 기능을 빠르게 테스트할 수 있음
- 파랑색이 generation한 부분
    
    ![이미지 0829146.jpg](/assets/HF_C1/이미지 0829146.jpg)
    

### (6) 파이프라인 종류 3- Mask filling

- mask filling은 주어진 텍스트의 공백을 채우는 것

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

- `top_k` →  출력할 공백 채우기 종류의 개수를 지정
- *mask token* 이라고 부르는 특수한 <mask> 단어를 채움
- 마스크 채우기(mask-filling) 모델에 따라 서로 다른 마스크 토큰을 요구할 수 있으므로 다른 모델을 탐색할 때 항상 해당 마스크 토큰을 확인하는 것이 좋음
- 위젯에서 사용된 부분 보고 확인 가능함

![이미지 0829147.jpg](/assets/HF_C1/이미지 0829147.jpg)

- 해당 사례에서는 mathematical, computational로 filling함

### (7) 파이프라인 종류 4- Named entity recognition 개체명 인식

- 개체명 인식(NER, Named Entity Recognition)은 입력 텍스트에서 어느 부분이 사람, 위치 또는 조직과 같은 개체명에 해당하는지 식별하는 작업

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

![이미지 0829148.jpg](/assets/HF_C1/이미지 0829148.jpg)

- 여기서 모델은 "Sylvain"이 사람(PER)이고 "Hugging Face"가 조직(ORG)이며 "Brooklyn"이 위치(LOC)으로 식별함
- 파이프라인 생성 함수에서 `grouped_entities=True` → 파이프라인이 동일한 엔티티에 해당하는 문장의 부분(토큰 혹은 단어)들을 그룹화하도록 함
- 여기서 모델은 "Hugging"과 "Face"를 단일 조직(ORG)으로 올바르게 그룹화했지만 이름 자체는 여러 단어로 구성됨
- 전처리 과정에서 심지어 일부 단어를 더 작은 부분으로 나눌 수도 있음 ex) Sylvain은 S, ##yl, ##va 및 ##in의 네 부분으로 나뉨
- 후처리 단계에서 파이프라인은 해당 조각을 성공적으로 재그룹화하여, "Sylvain"이 단일 단어로 출력됨

### (8) 파이프라인 종류 5- ****Question Answering****

- `question-answering` 은 주어진 context 정보를 사용하여 입력 질문에 응답을 제공함

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
```

![이미지 0829149.jpg](/assets/HF_C1/이미지 0829149.jpg)

- 제공된 context에서 정보를 추출하여 응답을 제공함

### (9) 파이프라인 종류 6 - Summarization

- summarization은 텍스트에 존재하는 중요한 내용을 유지하면서 해당 텍스트를 더 짧은 텍스트로 줄이는 작업임
- `max_length` 또는 `min_length` 지정 가능

```python
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer(
    """
Remembering that I'll be dead soon is the most important tool I've ever encountered to help me make the big choices in life. Because almost everything — all external expectations, all pride, all fear of embarrassment or failure - these things just fall away in the face of death, leaving only what is truly important. Remembering that you are going to die is the best way I know to avoid the trap of thinking you have something to lose. You are already naked. There is no reason not to follow your heart.
About a year ago I was diagnosed with cancer. I had a scan at 7:30 in the morning, and it clearly showed a tumor on my pancreas. I didn't even know what a pancreas was. The doctors told me this was almost certainly a type of cancer that is incurable, and that I should expect to live no longer than three to six months. My doctor advised me to go home and get my affairs in order, which is doctor's code for prepare to die. It means to try to tell your kids everything you thought you'd have the next 10 years to tell them in just a few months. It means to make sure everything is buttoned up so that it will be as easy as possible for your family. It means to say your goodbyes.
I lived with that diagnosis all day. Later that evening I had a biopsy, where they stuck an endoscope down my throat, through my stomach and into my intestines, put a needle into my pancreas and got a few cells from the tumor. I was sedated, but my wife, who was there, told me that when they viewed the cells under a microscope the doctors started crying because it turned out to be a very rare form of pancreatic cancer that is curable with surgery. I had the surgery and I'm fine now.
This was the closest I've been to facing death, and I hope its the closest I get for a few more decades. Having lived through it, I can now say this to you with a bit more certainty than when death was a useful but purely intellectual concept:
No one wants to die. Even people who want to go to heaven don't want to die to get there. And yet death is the destination we all share. No one has ever escaped it. And that is as it should be, because Death is very likely the single best invention of Life. It is Life's change agent. It clears out the old to make way for the new. Right now the new is you, but someday not too long from now, you will gradually become the old and be cleared away. Sorry to be so dramatic, but it is quite true.
Your time is limited, so don't waste it living someone else's life. Don't be trapped by dogma — which is living with the results of other people's thinking. Don't let the noise of others' opinions drown out your own inner voice. And most important, have the courage to follow your heart and intuition. They somehow already know what you truly want to become. Everything else is secondary.
When I was young, there was an amazing publication called The Whole Earth Catalog, which was one of the bibles of my generation. It was created by a fellow named Stewart Brand not far from here in Menlo Park, and he brought it to life with his poetic touch. This was in the late 1960's, before personal computers and desktop publishing, so it was all made with typewriters, scissors, and polaroid cameras. It was sort of like Google in paperback form, 35 years before Google came along: it was idealistic, and overflowing with neat tools and great notions.
Stewart and his team put out several issues of The Whole Earth Catalog, and then when it had run its course, they put out a final issue. It was the mid-1970s, and I was your age. On the back cover of their final issue was a photograph of an early morning country road, the kind you might find yourself hitchhiking on if you were so adventurous. Beneath it were the words: "Stay Hungry. Stay Foolish." It was their farewell message as they signed off. Stay Hungry. Stay Foolish. And I have always wished that for myself. And now, as you graduate to begin anew, I wish that for you.
Stay Hungry. Stay Foolish.
Thank you all very much.
 """
)
```

![이미지 0829150.jpg](/assets/HF_C1/이미지 0829150.jpg)

- 스티브잡스 연설문 일부를 가지고 와서 요약함
- There is no reason not to follow your heart

### (10) 파이프라인 종류 7 - Translation

- Translation의 경우 task 이름에 언어 pair를 지정하면 ex) "`translation_en_to_fr`default model을 사용할 수 있음
- [Model Hub](https://huggingface.co/models)에서 사용하려는 모델을 선택하는 방법도 있음 아래는 프랑스어 → 영어 번역임

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```

- `max_length` 와 `min_length` 지정 가능

---

지금까지의 pipeline은 demo용으로 specific tasks를 위한 복잡한 작업은 수행 되지 않음

앞으로는 pipeline() 함수를 어떻게 변형해서 사용할지에 대해 알아보려고 함 

그리고 아래는 pipeline 사용에 대한 document임

[Pipelines (huggingface.co)](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.FillMaskPipeline)

# References

### 강의 자료

[한글위키](https://wikidocs.net/book/8056)

[강의링크](https://huggingface.co/learn/nlp-course/chapter1/1)

### colab 파일

[https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter1/section3.ipynb#scrollTo=2gf_azYbKhta](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter1/section3.ipynb#scrollTo=2gf_azYbKhta)