

```
layout: post
title:  "심부전증 사망자 예측"
summary: "heart disease"
author: KSJ
date: '2021-03-08 09:41:00 +0900'
categories: analytics

```





***

심부전증 데이터를 활용하여 DATA HANDLING, EDA, MODELING을 진행해 보려고 합니다.

-----

![심부전증](/assets/brain-3017071_1920.png)



## Contents

1. [데이터 분석 문제 정의](#1.-데이터-분석-문제-정의)
2. [데이터 핸들링](#2.데이터-핸들링)
3. [데이터 EDA](3.-데이터EDA)
4. [모델링](#4.-모델링)
5. [마무리](#5.-마무리)



***

## 1. 데이터 분석 문제 정의

핸들링에 필요한 파이썬 라이브러리인 `numpy`, `pandas` 등의 기본 라이브러리를 import하고   train/test데이터를 업로드한다.

{% highlight python %}

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

{% endhighlight %}



## 3.데이터 EDA

### - 컬럼별 EDA

{% highlight python %}

\# pd.read_csv()로 csv파일 읽어들이기

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

{% endhighlight %}  



{% highlight python %}

df.head() # -5하면 앞 5개/뒤의 5개까지 함께

{% endhighlight %}  



```
|    |   age |   anaemia |   creatinine_phosphokinase |   diabetes |   ejection_fraction |   high_blood_pressure |   platelets |   serum_creatinine |   serum_sodium |   sex |   smoking |   time |   DEATH_EVENT |
|---:|------:|----------:|---------------------------:|-----------:|--------------------:|----------------------:|------------:|-------------------:|---------------:|------:|----------:|-------:|--------------:|
|  0 |    75 |         0 |                        582 |          0 |                  20 |                     1 |      265000 |                1.9 |            130 |     1 |         0 |      4 |             1 |
|  1 |    55 |         0 |                       7861 |          0 |                  38 |                     0 |      263358 |                1.1 |            136 |     1 |         0 |      6 |             1 |
|  2 |    65 |         0 |                        146 |          0 |                  20 |                     0 |      162000 |                1.3 |            129 |     1 |         1 |      7 |             1 |
|  3 |    50 |         1 |                        111 |          0 |                  20 |                     0 |      210000 |                1.9 |            137 |     1 |         0 |      7 |             1 |
|  4 |    65 |         1 |                        160 |          1 |                  20 |                     0 |      327000 |                2.7 |            116 |     0 |         0 |      8 |             1 |
```



{% highlight python %}

\#  df.info: 데이터타입/non-null count 

df.info()

\# 해석: 모든 데이터가 비워있지 않은 상태

{% endhighlight %}  

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



{% highlight python %}

\# df.describe(): 수치형 데이터의 통계

df.describe()

\# 해석: 0과 1로 이뤄진 변수의 mean을 보면 imbalance 상태를 볼 수 있음

\# max min이 과도한지 않은가/증가세가 일정한가

\# 시간과 사망은 관련이 있을 것

{% endhighlight %}  



```
|       |      age |    anaemia |   creatinine_phosphokinase |   diabetes |   ejection_fraction |   high_blood_pressure |   platelets |   serum_creatinine |   serum_sodium |        sex |   smoking |     time |   DEATH_EVENT |
|:------|---------:|-----------:|---------------------------:|-----------:|--------------------:|----------------------:|------------:|-------------------:|---------------:|-----------:|----------:|---------:|--------------:|
| count | 299      | 299        |                    299     | 299        |            299      |            299        |       299   |          299       |      299       | 299        | 299       | 299      |     299       |
| mean  |  60.8339 |   0.431438 |                    581.839 |   0.41806  |             38.0836 |              0.351171 |    263358   |            1.39388 |      136.625   |   0.648829 |   0.32107 | 130.261  |       0.32107 |
| std   |  11.8948 |   0.496107 |                    970.288 |   0.494067 |             11.8348 |              0.478136 |     97804.2 |            1.03451 |        4.41248 |   0.478136 |   0.46767 |  77.6142 |       0.46767 |
| min   |  40      |   0        |                     23     |   0        |             14      |              0        |     25100   |            0.5     |      113       |   0        |   0       |   4      |       0       |
| 25%   |  51      |   0        |                    116.5   |   0        |             30      |              0        |    212500   |            0.9     |      134       |   0        |   0       |  73      |       0       |
| 50%   |  60      |   0        |                    250     |   0        |             38      |              0        |    262000   |            1.1     |      137       |   1        |   0       | 115      |       0       |
| 75%   |  70      |   1        |                    582     |   1        |             45      |              1        |    303500   |            1.4     |      140       |   1        |   1       | 203      |       1       |
| max   |  95      |   1        |                   7861     |   1        |             80      |              1        |    850000   |            9.4     |      148       |   1        |   1       | 285      |       1       |
```





### - 수치형 데이터 EDA 

#### seaborn의 histplot, jointplot, pairplot

{% highlight python %}

sns.histplot(x='age',data=df, hue='DEATH_EVENT', kde =True)

\# 해석: 롱테일의 구조를 가지고 있는 구조

\# hue 강력함: 두 개의 히스토그램으로 쪼개져있음 겹쳐있음

\# 사망한 사람은 나이대가 고루 분포 / 사망하지 않은 사람은 젊은 쪽으로 몰려있음

{% endhighlight %}  



{% highlight python %}

sns.histplot(x='creatinine_phosphokinase',data=df)

\#아웃라이어가 많음

{% endhighlight %}  



{% highlight python %}

sns.histplot(data =df.loc[df['creatinine_phosphokinase'] <3000,'creatinine_phosphokinase'])

\#통계적인 특성이 잘 드러나지 않음

{% endhighlight %}  



{% highlight python %}

\# 'ejection_fraction'

\#sns.histplot(data = df, x='ejection_fraction') # 중간에 빈 경우는 bins 다시 조정

sns.histplot(data = df, x='ejection_fraction',bins = 13,hue = 'DEATH_EVENT' ,kde =True )

\# 'ejection_fraction'이 낮은 사람이 사망을 많이 함

{% endhighlight %}  





{% highlight python %}

\#혈소판 : 전체 히스토그램은 통계적으로 보이는데, 도움이 안 될듯함 death이벤트와 상관이 없어보임

sns.histplot(data = df, x='platelets',bins = 13,hue = 'DEATH_EVENT' ,kde =True )

{% endhighlight %}  



{% highlight python %}

\# 조인트 플랏: 히스토그램이나 kde플랏을 보여주고, 스캐터플랏을 보여줌 

sns.jointplot(x='platelets',y='creatinine_phosphokinase',hue = 'DEATH_EVENT',data=df, alpha = 0.3)

\# 뭉쳐서 판단 어려울때 알파값 조절

\# 뭉쳐있어서 판단에 큰 도움은 되지 않을 듯함

{% endhighlight %}  



### - 범주형 데이터 EDA

####  Boxplot 계열

{% highlight python %}

\# seaborn의 Boxplot 계열(boxplot(), violinplot(), swarmplot())을 사용

\# Hint) hue 키워드를 사용하여 범주 세분화 가능



\# 범주형은 박스플롯으로 봄

sns.boxplot(x='DEATH_EVENT',y='ejection_fraction',data=df)

{% endhighlight %}  



{% highlight python %}

sns.boxplot(x='smoking',y='ejection_fraction',data=df)

\# 흡연자의 ejection_fraction이 좁음

{% endhighlight %}  



{% highlight python %}

sns.violinplot(x='DEATH_EVENT',y='ejection_fraction',data=df,hue='smoking')

\#박스플롯의 변형 -> 박스플랏 + 히스토그램 정보 + 아웃라이어 (보고할때는 박스플랏이 더 나음)

{% endhighlight %}  



{% highlight python %}

sns.swarmplot(x='DEATH_EVENT',y='ejection_fraction',data=df,hue='smoking')

\#스캐터 + 바이올린 플랏 합친 것 대신 박스플랏의 통계정보는 없음

{% endhighlight %}  







