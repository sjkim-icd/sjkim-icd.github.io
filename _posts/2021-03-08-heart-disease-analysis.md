---
layout: post
title:  "의료데이터: 심부전증 사망자 예측"
summary: "heart disease"
author: KSJ
date: '2021-03-08 09:41:00 +0900'
categories: analytics
---

***

심부전증 데이터를 활용하여 DATA HANDLING, EDA, MODELING을 진행해 보려고 합니다.

-----

![심부전증](/assets/brain-3017071_1920.png)

의료데이터인 심부전증 데이터를 활용하여 DATA HANDLING, EDA, MODELING을 진행해 보려고 합니다.

## Contents

1. [데이터 분석 문제 정의](#1.-데이터-분석-문제-정의)
2. [데이터 EDA](3.-데이터EDA)
3. [데이터 핸들링](#2.데이터-핸들링)
4. [모델링](#4.-모델링)
5. [마무리](#5.-마무리)



***

## 1. 데이터 분석 문제 정의

**target의 값은 death_event**이며, 심부전증으로 인한 사망을 예측하는 것이 해당 데이터의 분석 목적입니다.

***

핸들링에 필요한 파이썬 라이브러리인 `numpy`, `pandas` 등의 기본 라이브러리를 import하고   train/test데이터를 업로드합니다.

{% highlight python %}

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

{% endhighlight %}

## 2.데이터 EDA

### 2-1. 컬럼별 EDA

#### pd.read_csv()로 csv파일 읽어들이기

{% highlight python %}

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

{% endhighlight %}  

###### 

***

#### df.head(): (-5)의 경우 head/tail 5 출력

{% highlight python %}

df.head() 

|      |  age | anaemia | creatinine_phosphokinase | diabetes | ejection_fraction | high_blood_pressure | platelets | serum_creatinine | serum_sodium |  sex | smoking | time | DEATH_EVENT |
| ---: | ---: | :------ | -----------------------: | -------: | ----------------: | ------------------: | --------: | ---------------: | -----------: | ---: | ------: | ---: | ----------: |
|    0 |   75 | 0       |                      582 |        0 |                20 |                   1 |    265000 |              1.9 |          130 |    1 |       0 |    4 |           1 |
|    1 |   55 | 0       |                     7861 |        0 |                38 |                   0 |    263358 |              1.1 |          136 |    1 |       0 |    6 |           1 |
|    2 |   65 | 0       |                      146 |        0 |                20 |                   0 |    162000 |              1.3 |          129 |    1 |       1 |    7 |           1 |
|    3 |   50 | 1       |                      111 |        0 |                20 |                   0 |    210000 |              1.9 |          137 |    1 |       0 |    7 |           1 |
|    4 |   65 | 1       |                      160 |        1 |                20 |                   0 |    327000 |              2.7 |          116 |    0 |       0 |    8 |           1 |

{% endhighlight %}  df.info(): 데이터의 타입/non-null count 

{% highlight python %}

df.info()

{% endhighlight %}  

모든 컬럼이 non-null이며, 즉 null이 없는 상태입니다.

```
#   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   age                       299 non-null    float64
 1   anaemia                   299 non-null    int64  
 2   creatinine_phosphokinase  299 non-null    int64  
 3   diabetes                  299 non-null    int64  
 4   ejection_fraction         299 non-null    int64  
 5   high_blood_pressure       299 non-null    int64  
 6   platelets                 299 non-null    float64
 7   serum_creatinine          299 non-null    float64
 8   serum_sodium              299 non-null    int64  
 9   sex                       299 non-null    int64  
 10  smoking                   299 non-null    int64  
 11  time                      299 non-null    int64  
 12  DEATH_EVENT               299 non-null    int64  
```

***

#### df.describe(): 수치형 데이터의 통계

{% highlight python %}

df.describe()

|       |     age |  anaemia | creatinine_phosphokinase | diabetes | ejection_fraction | high_blood_pressure | platelets | serum_creatinine | serum_sodium |      sex | smoking |    time | DEATH_EVENT |
| :---- | ------: | -------: | -----------------------: | -------: | ----------------: | ------------------: | --------: | ---------------: | -----------: | -------: | ------: | ------: | ----------: |
| count |     299 |      299 |                      299 |      299 |               299 |                 299 |       299 |              299 |          299 |      299 |     299 |     299 |         299 |
| mean  | 60.8339 | 0.431438 |                  581.839 |  0.41806 |           38.0836 |            0.351171 |    263358 |          1.39388 |      136.625 | 0.648829 | 0.32107 | 130.261 |     0.32107 |
| std   | 11.8948 | 0.496107 |                  970.288 | 0.494067 |           11.8348 |            0.478136 |   97804.2 |          1.03451 |      4.41248 | 0.478136 | 0.46767 | 77.6142 |     0.46767 |
| min   |      40 |        0 |                       23 |        0 |                14 |                   0 |     25100 |              0.5 |          113 |        0 |       0 |       4 |           0 |
| 25%   |      51 |        0 |                    116.5 |        0 |                30 |                   0 |    212500 |              0.9 |          134 |        0 |       0 |      73 |           0 |
| 50%   |      60 |        0 |                      250 |        0 |                38 |                   0 |    262000 |              1.1 |          137 |        1 |       0 |     115 |           0 |
| 75%   |      70 |        1 |                      582 |        1 |                45 |                   1 |    303500 |              1.4 |          140 |        1 |       1 |     203 |           1 |
| max   |      95 |        1 |                     7861 |        1 |                80 |                   1 |    850000 |              9.4 |          148 |        1 |       1 |     285 |           1 |

{% endhighlight %}  

이 describe에서 알 수 있는 것은 다음과 같습니다.

1) 0과 1로 이뤄진 변수의 mean을 보면 imbalance 상태를 알아볼 수 있습니다.

2) max min이 과도한지 않은가/증가세가 일정한가를 통해 outlier 등을 파악할 수 있습니다.2-2. 수치형 데이터 EDA 

#### seaborn의 histplot, jointplot, pairplot

수치형 데이터의 EDA는 seaborn의 histplot, jointplot을 통해 진행할 수 있습니다.

#### sns.histplot

- AGE

{% highlight python %}

sns.histplot(x='age',data=df, hue='DEATH_EVENT', kde =True)

{% endhighlight %}  

![hist](/assets/kaggle/심부전증/hisplot.png)

histplot을 살펴보면 롱테일의 구조를 가지고 있습니다.

hisplot의 hue는 강력한데요. hue에 따라 여러개의 히스토그램으로 쪼개져서 표현되고 겹치는 부분은 회색으로 표현되네요.

사망한 사람은 나이대가 고루게 분포하고 사망하지 않은 사람은 젊은 쪽으로 몰려있음을 알 수 있습니다.



- creatinine_phosphokinase

먼저, 그래프를 그려보고 outlier가 많은 경우 특정 값 이하만 추출하여

다시 그래프를 그렸습니다.

{% highlight python %}

sns.histplot(data =df.loc[df['creatinine_phosphokinase'] <3000,'creatinine_phosphokinase'])

{% endhighlight %}  

![hist1](/assets/kaggle/심부전증/hist2.png)

해당 그래프에서는 딱히 통계적인 특성이 드러나지는 않네요.



- ejection_fraction

  histogram 중간에 비는 경우, bins가 좁아 그런 경우도 있으니, bins를 다시 조정해봅니다.

{% highlight python %}

\#sns.histplot(data = df, x='ejection_fraction') 

sns.histplot(data = df, x='ejection_fraction',bins = 13,hue = 'DEATH_EVENT' ,kde =True )

{% endhighlight %}  

![hist3](/assets/kaggle/심부전증/hist3.png)

 'ejection_fraction'이 낮은 사람이 사망을 많이 하는 경향을 볼 수 있습니다.



- platelets 혈소판

{% highlight python %}

sns.histplot(data = df, x='platelets',bins = 13,hue = 'DEATH_EVENT' ,kde =True )

{% endhighlight %}  

![hist4](/assets/kaggle/심부전증/hist4.png)

혈소판의 경우, 전체 히스토그램은 통계적으로 보이는데,

 death이벤트와 상관성은 보이지 않는 듯 합니다.



#### joint plot :히스토그램,KDE플랏, SCATTER 플랏

{% highlight python %}

sns.jointplot(x='platelets',y='creatinine_phosphokinase',hue = 'DEATH_EVENT',data=df, alpha = 0.3)

{% endhighlight %}  

![hist5](/assets/kaggle/심부전증/hist5.png)

스캐터플랏이 뭉쳐서 판단 어려울때 알파값을 조절하여

투명도를 조정합니다. 대부분의 점들이 뭉쳐있어

판단을 내리기는 힘들어 보입니다.



### 2-3. 범주형 데이터 EDA

####  Boxplot 계열

범주형은 박스플롯 계열의 boxplot(), violinplot(), swarmplot()을 사용합니다. hue 키워드를 사용하여 범주 세분화가 가능합니다.

#### sns.boxplot

{% highlight python %}

sns.boxplot(x='DEATH_EVENT',y='ejection_fraction',data=df)

{% endhighlight %}  

![hist6](/assets/kaggle/심부전증/hist6.png)

박스플랏으로 통계치를 한 눈에 보고, 아웃라이어도 볼 수 있습니다.

박스플랏의 경우, 간단하지만 여러 정보가 담겨있어 경영층과 대화할 때 많이 쓰곤합니다.

#### sns.violinplot

violinplot은 박스플롯의 변형으로 박스플랏 + 히스토그램 정보 + 아웃라이어의 정보를 담고 있습니다.

{% highlight python %}

sns.violinplot(x='DEATH_EVENT',y='ejection_fraction',data=df,hue='smoking')

{% endhighlight %}  

![hist7](/assets/kaggle/심부전증/hist7.png)

#### swarmplot

swarmplot은 스캐터 , 바이올린 플랏을 합친 것으로 볼 수 있는데요. 대신 박스플랏의 통계정보는 표시되지 않습니다.

{% highlight python %}

sns.swarmplot(x='DEATH_EVENT',y='ejection_fraction',data=df,hue='smoking')

{% endhighlight %}  

![hist8](/assets/kaggle/심부전증/hist8.png)



## 





