---
title: "[Hugging Face][C-2] Handling multiple sequences"
header:
  overlay_filter: 0.5

categories:
  - NLP
tags:
  - [Hugging Handling multiple sequences"]
comments: true
toc: true
toc_sticky: true
 
date: 2023-09-17
last_modified_at: 2023-09-17
---


# Overview

## ****Batching inputs together with Sylvain****

![이미지 0906022.jpg](/assets/HF/이미지 906022.jpg)

- 입력 시퀀스를 일괄 처리하는 방법에 대해서 살펴보고자 함
- 일반적으로 모델을 통해 전달하려는 문장의 길이는 모두 동일하지 않음
    
    ![이미지 0906023.jpg](/assets/HF/이미지 906023.jpg)
    
- 여기서는 sentiment analysis pipeline에서 본 모델을 사용하여 두 문장을 분류하려고 함
- 이를 토큰화하고 각 토큰을 해당 입력 ID에 매핑하면 길이가 다른 두 개의 목록이 생성됨
    
    ![이미지 0906024.jpg](/assets/HF/이미지 906024.jpg)
    
    ![이미지 0906025.jpg](/assets/HF/이미지 906025.jpg)
    
- 모든 배열과 텐서는 직사각형이어야 하기 때문에 이 두 목록에서 텐서 또는 Numpy 배열을 만들려고 하면 오류가 발생함
    
    ![이미지 0906026.jpg](/assets/HF/이미지 906026.jpg)
    
- 이를 극복하기 위한 방법은 special token을 필요한 만큼 추가하여 두 번째 문당을 첫 번쨰 문장과 동일한 길이로 만드는 것임
- 또 다른 방법은 첫 번째 시퀀스를 두 번째 시퀀스의 길이로 자르는 것이지만 문장을 적절하게 분류하는데 필요한 많은 정보를 잃게 됨
- 일반적으로 문장이 모델이 처리할 수 있는 최대 길이보다 긴 경우에만 문장을 자름
- 두 번째 문장을 채우는 데 사용되는 값은 무작위로 선택해서는 안됨
    
    ![이미지 0906027.jpg](/assets/HF/이미지 906027.jpg)
    
- 모델은 tokenizer.pad_token_id에서 찾을 수 있는 특정 패딩 ID로 사전학습됨
- 문장을 채웠으므로 일괄 처리를 할 수 있음
    
    ![이미지 0906028.jpg](/assets/HF/이미지 906028.jpg)
    
- 하지만 두 문장을 별도로 모델에 전달하고 함께 일괄 처리하면 패딩된 문장(여기서는 두번째 문장)에 대해 동일한 결과를 얻지 못함을 알 수 있음
- Transformer 모델이 어텐션 레이어를 많이 사용한다는 점을 기억한다면 이는 놀랄일이 아님
- 각 토큰의 문맥 표현을 계산할 때 attention layer는 문장의 다른 모든 단어를 봄
    
    ![이미지 0906029.jpg](/assets/HF/이미지 906029.jpg)
    
- 문장만 있거나 여러 패딩 토큰이 추가된 문장이 있는 경우, 동일한 값을 얻지 못하는 것이 논리적임
    
    ![이미지 0906030.jpg](/assets/HF/이미지 906030.jpg)
    
- 패딩이 있든 없든 동일한 결과를 얻으려면 attention layer에 해당 패딩 토큰을 무시해야 함을 알려줘야 함
- 이는 입력ID와 모양이 동일하고 0과 1이 있는 텐서인 attention mask를 생성하여 수행함
- 1은 attention layer가 context에서 고려해야 하는 토큰을 나타내고 0은 무시해야하는 토큰을 나타냄
    
    ![이미지 0906031.jpg](/assets/HF/이미지 906031.jpg)
    
- 이제 입력 ID와 함께 이 attention mask를 전달하면 두 문장을 모델에 개별적으로 보낼 때와 동일한 결과를 얻을 수 있음
    
    ![이미지 0906032.jpg](/assets/HF/이미지 906032.jpg)
    
- 이 모든 작업은 패딩 = Ture 플래그를 사용하여 여러 문장에 적용할 때 토크나이저에 의해 백그라운드에서 수행됨
- 작은 문장에 적절한 값의 패딩을 적용하고 적절한 attention mask를 생성함
- 이전 섹션에서 우리는 가장 간단한 활용 사례를 살펴보았음
- 즉, 길이가 짧은 단일 시퀀스에 대한 추론을 수행하는 것
- 그러나 몇 가지 의문점이 존재함
    - 다중 시퀀스(multiple sequences)를 어떻게 처리할까?
    - 각각이 길이가 다른 여러 개의 시퀀스를 어떻게 처리할까?
    - 모델이 잘 동작할 수 있게 하기 위해서 어휘집(vocabulary)의 인덱스들만 입력하면 될까?
    - 길이가 엄청나게 긴 시퀀스에 대해서는 잘 처리할 수 있을까?
- 위 질문들이 어떤 문제를 어떻게 해결하는지를 보고 
Transformers API를 사용하여 해결할 수 있는지 알아보고자 함

# ****Models expect a batch of inputs****

- 모델(model)은 입력의 배치(batch) 형태를 요구함
- 이전 섹션에서 시퀀스가 숫자 리스트로 변환되는 방법을 보았음.
- 이 숫자 리스트를 텐서(tensor)로 변환하고 모델에 입력해 보고자 함

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)
# This line will fail
model(input_ids)
```

```python
---------------------------------------------------------------------------

IndexError                                Traceback (most recent call last)

/tmp/ipykernel_9651/1126667217.py in <module>
     12 input_ids = torch.tensor(ids)
     13 # This line will fail
---> 14 model(input_ids)

~/anaconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1101                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1102             return forward_call(*input, **kwargs)
   1103         # Do not call functions when jit is used
   1104         full_backward_hooks, non_full_backward_hooks = [], []

~/anaconda3/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
    727         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    728 
--> 729         distilbert_output = self.distilbert(
    730             input_ids=input_ids,
    731             attention_mask=attention_mask,

~/anaconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1101                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1102             return forward_call(*input, **kwargs)
   1103         # Do not call functions when jit is used
   1104         full_backward_hooks, non_full_backward_hooks = [], []

~/anaconda3/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    548 
    549         if inputs_embeds is None:
--> 550             inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
    551         return self.transformer(
    552             x=inputs_embeds,

~/anaconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1101                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1102             return forward_call(*input, **kwargs)
   1103         # Do not call functions when jit is used
   1104         full_backward_hooks, non_full_backward_hooks = [], []

~/anaconda3/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids)
    117         embeddings)
    118         """
--> 119         seq_length = input_ids.size(1)
    120 
    121         # Setting the position-ids to the registered buffer in constructor, it helps

IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
```

- 위 코드에서 우리는 섹션 2에서의 파이프라인 단계를 그대로 따랐는데 오류가 생김
- 모델에 하나의 단일 시퀀스를 입력해서 발생하는 문제임
- Transformers 모델은 기본적으로 다중 문장(시퀀스)을 한번에 입력해야 함
- 시퀀스에 토크나이저를 적용할 때 실제 내부적으로 수행하는 모든 작업을 시도함
- 하지만 아래 코드를 자세히 보면 입력 식별자(input IDs) 리스트를 텐서로 변환하는 동시에 차원(dimension) 하나가 그 위에 추가되는 것을 알 수 있음

```python
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
```

```python
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102]])
```

- 오류가 발생한 코드에서 `input_ids`에 새로운 차원을 하나 추가해 보고자 함

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
```

```python
Input IDs: tensor([[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,
          2026,  2878,  2166,  1012]])
Logits: tensor([[-2.7276,  2.8789]], grad_fn=<AddmmBackward0>)
```

- 위 코드에서는 입력 식별자(input IDs)와 그 결과 로짓(logit) 값을 출력하고 있음
- *Batching* 이란 모델을 통해 한번에 여러 문장을 입력하는 동작임
- 문장이 하나만 있는 경우 아래와 같이 단일 시퀀스로 배치(batch) 를 빌드할 수 있음

```python
batch_ids = [ids, ids]
```

- 동일한 두 시퀀스로 구성된 배치(batch)임
- 배치(batch) 처리를 통해서 모델이 여러 문장을 동시에 입력받을 수 있도록 할 수 있음
- 다중 시퀀스를 사용하는 것은 단일 시퀀스로 배치(batch)를 구성하는 것만큼 간단함
- 하지만 두 번째 문제가 있음. 두 개(또는 그 이상) 문장을 함께 배치(batch) 처리하려고 할 때 각 문장의 길이가 다를 수 있음
- 텐서(tensor)는 형태가 직사각형 모양이어야 함
- 따라서 이럴 경우에는 입력 식별자(input IDs) 리스트를 텐서로 직접 변환할 수 없음
- 이 문제를 해결하기 위해 일반적으로 입력을 *padding으로 채움*

# ****Padding the inputs****

- 다음 리스트의 리스트(혹은 이중 리스트)는 텐서로 변환할 수 없음

```python
batched_ids = [
    [200, 200, 200],
    [200, 200],
]
```

- 이 문제를 해결하기 위해 *패딩(padding)* 을 사용하여 텐서를 직사각형 모양으로 만듦
- 패딩(padding)은 길이가 더 짧은 문장에 *패딩 토큰(padding token)* 이라는 특수 단어를 추가하여 모든 문장이 동일한 길이를 갖도록 함
- 예를 들어, 10개의 단어로 구성된 10개의 문장과 20개의 단어가 있는 1개의 문장이 있는 경우, 패딩(padding)을 사용하면 모든 문장에 20개의 단어가 포함됨. 위의 `batched_ids`를 패딩(padding) 처리하면 결과 텐서는 다음과 같음

```python
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```

- 패딩 토큰(padding token)의 식별자(ID)는 `tokenizer.pad_token_id`에 지정되어 있음. 이를 활용하여 두 개의 시퀀스를 한번은 개별적으로 또 한번은 배치(batch) 형태로 모델에 입력해 보고자 함

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
```

```python
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
tensor([[ 1.5694, -1.3895],
        [ 1.3374, -1.2163]], grad_fn=<AddmmBackward0>)
```

- 배치 처리된 예측 결과의 로짓(logits)에 문제가 있음.
- 두 번째 행은 두 번째 문장의 로짓(logits)과 같아야 하지만 완전히 다른 값을 가짐
- 이는 트랜스포머(Transformer) 모델의 핵심적인 특징이 각 토큰을 *컨텍스트화(contextualize)* 하는 어텐션 레이어(attention layers)를 가지고 있다는 사실이기 때문임
- 어텐션 레이어(attention layers)는 시퀀스의 모든 토큰에 주의 집중(paying attention)을 하기 때문에 패딩 토큰도 역시 고려함
- 모델에 길이가 다른 개별 문장들을 입력할 때나 동일한 문장으로 구성된 패딩이 적용된 배치(batch)를 입력할 때 동일한 결과를 얻기 위해서는 해당 어텐션 레이어(attention layers)가 패딩 토큰을 무시하도록해야함
- 이는 어텐션 마스크(attention mask)를 사용하여 처리할 수 있음

# ****Attention masks****

- 어텐션 마스크(attention mask)는 0과 1로 채워진 입력 식별자(input IDs) 텐서(tensor)와 형태가 정확하게 동일한 텐서(tensor)임
- 1은 해당 토큰에 주의를 기울여야 함을 나타내고 0은 해당 토큰을 무시해야 함을 나타냄
- 즉, 모델의 어텐션 레이어(attention layers)에서 무시해야 함

```python
batch_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batch_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

```python
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
```

- 이제 배치(batch)의 두 번째 문장에 대해 동일한 로짓(logits) 값을 얻을 수 있음
- 두 번째 시퀀스의 마지막 값이 패딩 식별자(padding ID)이고 이에 해당하는 어텐션 마스크(attention mask)의 값이 0임

# ****Longer sequences****

- 트랜스포머(Transformer) 모델을 사용할 때, 모델에 입력할 수 있는 시퀀스의 길이에 제한이 있음
- 대부분의 모델은 최대 512개 또는 1024개의 토큰 시퀀스를 처리하며, 그보다 더 긴 시퀀스를 처리하라는 요청을 받으면 오류를 발생시는데이에 대한 두 가지 솔루션이 있음
    - 길이가 더 긴 시퀀스를 지원하는 모델 사용
    - 시퀀스 절단(truncation)
- 모델 별로 지원되는 시퀀스 길이가 다르며 일부 모델은 매우 긴 시퀀스 처리에 특화되어 있음.
- [Longformer](https://huggingface.co/transformers/model_doc/longformer.html)가 하나의 예이고 다른 하나는 [LED](https://huggingface.co/transformers/model_doc/led.html)
- 매우 긴 시퀀스를 필요로 하는 태스크를 수행하는 경우 해당 모델을 살펴보는 것이 좋음
- 그렇지 않으면, `max_sequence_length` 매개변수를 지정하여 시퀀스를 절단하는 것이 좋음

```python
max_sequence_length = 512

sequence = sequence[:max_sequence_length]
```