---
layout: post
title:  "의료데이터 예제"
summary: "heart disease"
author: KSJ
date: '2021-03-08 09:41:00 +0900'
categories: analytics

---



# 주제 : 데이터 분석으로 심부전증을 예방할 수 있을까?

----------

## 실습 가이드
    1. 데이터를 다운로드하여 Colab에 불러옵니다.
    2. 필요한 라이브러리는 모두 코드로 작성되어 있습니다.
    3. 코드는 위에서부터 아래로 순서대로 실행합니다.


​    
## 데이터 소개
    - 이번 주제는 Heart Failure Prediction 데이터셋을 사용합니다.
    
    - 다음 1개의 csv 파일을 사용합니다.
    heart_failure_clinical_records_dataset.csv
    
    - 각 파일의 컬럼은 아래와 같습니다.
    age: 환자의 나이
    anaemia: 환자의 빈혈증 여부 (0: 정상, 1: 빈혈)
    creatinine_phosphokinase: 크레아틴키나제 검사 결과
    diabetes: 당뇨병 여부 (0: 정상, 1: 당뇨)
    ejection_fraction: 박출계수 (%)
    high_blood_pressure: 고혈압 여부 (0: 정상, 1: 고혈압)
    platelets: 혈소판 수 (kiloplatelets/mL)
    serum_creatinine: 혈중 크레아틴 레벨 (mg/dL)
    serum_sodium: 혈중 나트륨 레벨 (mEq/L)
    sex: 성별 (0: 여성, 1: 남성)
    smoking: 흡연 여부 (0: 비흡연, 1: 흡연)
    time: 관찰 기간 (일)
    DEATH_EVENT: 사망 여부 (0: 생존, 1: 사망)


​    
​    
- 데이터 출처: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data


## 최종 목표
    - 의료 데이터와 그 분석에 대한 이해
    - Colab 및 Pandas 라이브러리 사용법 이해
    - 데이터 시각화를 통한 인사이트 습득 방법의 이해
    - Scikit-learn 기반의 모델 학습 방법 습득
    - Classification 모델의 학습과 평가 방법 이해

- 출제자 : 신제용 강사
---

## Step 0. 의료 데이터셋에 대하여

### 의료 데이터의 수집


### 의료 데이터 분석의 현재


### Accuracy, Precision, 그리고 Recall

## Step 1. 데이터셋 준비하기


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 문제 1. Colab Notebook에 Kaggle API 세팅하기



```python
import os
```


```python
# os.environ을 이용하여 Kaggle API Username, Key 세팅하기

os.environ['KAGGLE_USERNAME']='sojeongkimdesign'
os.environ['KAGGLE_KEY']='bd7482b7426c6d140c3e8ef3b602b202'

```

### 문제 2. 데이터 다운로드 및 압축 해제하기



```python
# Linux 명령어로 Kaggle API를 이용하여 데이터셋 다운로드하기 (!kaggle ~)
# Linux 명령어로 압축 해제하기
!kaggle -h
!kaggle datasets download -d andrewmvd/heart-failure-clinical-data
!unzip '*.zip'
```

    usage: kaggle [-h] [-v] {competitions,c,datasets,d,kernels,k,config} ...
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit
    
    commands:
      {competitions,c,datasets,d,kernels,k,config}
                            Use one of:
                            competitions {list, files, download, submit, submissions, leaderboard}
                            datasets {list, files, download, create, version, init, metadata, status}
                            config {view, set, unset}
        competitions        Commands related to Kaggle competitions
        datasets            Commands related to Kaggle datasets
        kernels             Commands related to Kaggle kernels
        config              Configuration settings
    Downloading heart-failure-clinical-data.zip to /content
      0% 0.00/3.97k [00:00<?, ?B/s]
    100% 3.97k/3.97k [00:00<00:00, 6.79MB/s]
    Archive:  heart-failure-clinical-data.zip
      inflating: heart_failure_clinical_records_dataset.csv  



```python
!ls # 현재 디렉토리에 있는 모든 파일
```

    heart-failure-clinical-data.zip		    sample_data
    heart_failure_clinical_records_dataset.csv


### 문제 3. Pandas 라이브러리로 csv파일 읽어들이기



```python
# pd.read_csv()로 csv파일 읽어들이기
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

```

## Step 2. EDA 및 데이터 기초 통계 분석


### 문제 4. 데이터프레임의 각 컬럼 분석하기



```python
# DataFrame에서 제공하는 메소드를 이용하여 컬럼 분석하기 (head(), info(), describe())
df.head(-5) # 뒤의 5개까지 함께
print(df.head().to_markdown())

```

    |    |   age |   anaemia |   creatinine_phosphokinase |   diabetes |   ejection_fraction |   high_blood_pressure |   platelets |   serum_creatinine |   serum_sodium |   sex |   smoking |   time |   DEATH_EVENT |
    |---:|------:|----------:|---------------------------:|-----------:|--------------------:|----------------------:|------------:|-------------------:|---------------:|------:|----------:|-------:|--------------:|
    |  0 |    75 |         0 |                        582 |          0 |                  20 |                     1 |      265000 |                1.9 |            130 |     1 |         0 |      4 |             1 |
    |  1 |    55 |         0 |                       7861 |          0 |                  38 |                     0 |      263358 |                1.1 |            136 |     1 |         0 |      6 |             1 |
    |  2 |    65 |         0 |                        146 |          0 |                  20 |                     0 |      162000 |                1.3 |            129 |     1 |         1 |      7 |             1 |
    |  3 |    50 |         1 |                        111 |          0 |                  20 |                     0 |      210000 |                1.9 |            137 |     1 |         0 |      7 |             1 |
    |  4 |    65 |         1 |                        160 |          1 |                  20 |                     0 |      327000 |                2.7 |            116 |     0 |         0 |      8 |             1 |



```python
# 중요: 데이터타입/non-null count 
df.info()
# 해석: 모든 데이터가 비워있지 않은 상태

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 299 entries, 0 to 298
    Data columns (total 13 columns):
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
    dtypes: float64(3), int64(10)
    memory usage: 30.5 KB



```python
# 수치형 데이터의 통계
df.describe()

# 해석: 0과 1로 이뤄진 변수의 mean을 보면 imbalance 상태를 볼 수 있음
# max min이 과도한지 않은가/증가세가 일정한가
# 시간과 사망은 관련이 있을 것

print(df.describe().to_markdown())
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


### 문제 5. 수치형 데이터의 히스토그램 그리기



```python
df.columns
```




    Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
           'DEATH_EVENT'],
          dtype='object')




```python
# seaborn의 histplot, jointplot, pairplot을 이용해 히스토그램 그리기

sns.histplot(x='age',data=df, hue='DEATH_EVENT', kde =True)
# 해석: 롱테일의 구조를 가지고 있는 구조
# hue 강력함: 두 개의 히스토그램으로 쪼개져있음 겹쳐있음
# 사망한 사람은 나이대가 고루 분포 / 사망하지 않은 사람은 젊은 쪽으로 몰려있음


```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fef0720df90>




![png](Chapter_01_%EC%8B%AC%EB%B6%80%EC%A0%84%EC%A6%9D_files/Chapter_01_%EC%8B%AC%EB%B6%80%EC%A0%84%EC%A6%9D_22_1.png)

![hist](/assets/kaggle/심부전증/Chapter_01_심부전증_22_1.png)

```python
sns.histplot(x='creatinine_phosphokinase',data=df)
#아웃라이어가 많음
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fef070b30d0>




![png](Chapter_01_%EC%8B%AC%EB%B6%80%EC%A0%84%EC%A6%9D_files/Chapter_01_%EC%8B%AC%EB%B6%80%EC%A0%84%EC%A6%9D_23_1.png)



```python
sns.histplot(data =df.loc[df['creatinine_phosphokinase'] <3000,'creatinine_phosphokinase'])
#통계적인 특성이 잘 드러나지 않음
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fef081ebb90>

![hist1](/assets/kaggle/심부전증/Chapter_01_심부전증_24_1.png)



```python
# 'ejection_fraction'
#sns.histplot(data = df, x='ejection_fraction') # 중간에 빈 경우는 bins 다시 조정
sns.histplot(data = df, x='ejection_fraction',bins = 13,hue = 'DEATH_EVENT' ,kde =True )
# 'ejection_fraction'이 낮은 사람이 사망을 많이 함
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fef0263fc10>




![png](Chapter_01_%EC%8B%AC%EB%B6%80%EC%A0%84%EC%A6%9D_files/Chapter_01_%EC%8B%AC%EB%B6%80%EC%A0%84%EC%A6%9D_25_1.png)



![hist1](/assets/kaggle/심부전증/Chapter_01_심부전증_25_1.png)

```python
#혈소판 : 전체 히스토그램은 통계적으로 보이는데, 도움이 안 될듯함 death이벤트와 상관이 없어보임
sns.histplot(data = df, x='platelets',bins = 13,hue = 'DEATH_EVENT' ,kde =True )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fef02409250>



![hist1](/assets/kaggle/심부전증/Chapter_01_심부전증_26_1.png)

```python
# 조인트 플랏: 히스토그램이나 kde플랏을 보여주고, 스캐터플랏을 보여줌 
sns.jointplot(x='platelets',y='creatinine_phosphokinase',hue = 'DEATH_EVENT',data=df, alpha = 0.3)
# 뭉쳐서 판단 어려울때 알파값 조절
# 뭉쳐있어서 판단에 큰 도움은 되지 않을 듯함
```




    <seaborn.axisgrid.JointGrid at 0x7feef9933cd0>



![hist1](/assets/kaggle/심부전증/Chapter_01_심부전증_27_1.png)


### 문제 6. Boxplot 계열을 이용하여 범주별 통계 확인하기





```python

```


```python
# seaborn의 Boxplot 계열(boxplot(), violinplot(), swarmplot())을 사용
# Hint) hue 키워드를 사용하여 범주 세분화 가능

# 범주형은 박스플롯으로 봄
sns.boxplot(x='DEATH_EVENT',y='ejection_fraction',data=df)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7feef97e7dd0>



![hist1](/assets/kaggle/심부전증/Chapter_01_심부전증_31_1.png)



```python
sns.boxplot(x='smoking',y='ejection_fraction',data=df)
# 흡연자의 ejection_fraction이 좁음
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7feef976f510>





![hist1](/assets/kaggle/심부전증/Chapter_01_심부전증_32_1.png)



```python
sns.violinplot(x='DEATH_EVENT',y='ejection_fraction',data=df,hue='smoking')
#박스플롯의 변형 -> 박스플랏 + 히스토그램 정보 + 아웃라이어 (보고할때는 박스플랏이 더 나음)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7feef97832d0>





![hist1](/assets/kaggle/심부전증/Chapter_01_심부전증_33_1.png)

```python
sns.swarmplot(x='DEATH_EVENT',y='ejection_fraction',data=df,hue='smoking')
#스캐터 + 바이올린 플랏 합친 것 대신 박스플랏의 통계정보는 없음
```

    /usr/local/lib/python3.7/dist-packages/seaborn/categorical.py:1296: UserWarning: 13.3% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
      warnings.warn(msg, UserWarning)





    <matplotlib.axes._subplots.AxesSubplot at 0x7fef02249110>



![hist1](/assets/kaggle/심부전증/Chapter_01_심부전증_34_2.png)


## Step 3. 모델 학습을 위한 데이터 전처리


### 문제 7. StandardScaler를 이용하여 데이터 전처리하기



```python
from sklearn.preprocessing import StandardScaler
```


```python
# 수치형 입력 데이터, 범주형 입력 데이터, 출력 데이터로 구분하기
X_num = 
X_cat = 
y = 
```


```python
# 수치형 입력 데이터를 전처리하고 입력 데이터 통합하기
scaler =
X = 
```

### 문제 8. 학습데이터와 테스트데이터 분리하기



```python
from sklearn.model_selection import train_test_split
```


```python
# train_test_split() 함수로 학습 데이터와 테스트 데이터 분리하기
X_train, X_test, y_train, y_test = 
```

## Step 4. Classification 모델 학습하기


### 문제 9. Logistic Regression 모델 생성/학습하기



```python
from sklearn.linear_model import LogisticRegression
```


```python
# LogisticRegression 모델 생성/학습
model_lr = 



```

### 문제 10. 모델 학습 결과 평가하기



```python
from sklearn.metrics import classification_report
```


```python
# Predict를 수행하고 classification_report() 결과 출력하기
pred = 



```

### 문제 11. XGBoost 모델 생성/학습하기



```python
from xgboost import XGBClassifier
```


```python
# XGBClassifier 모델 생성/학습
model_xgb = 



```

### 문제 12. 모델 학습 결과 평가하기



```python
# Predict를 수행하고 classification_report() 결과 출력하기
pred = 



```

### 문제 13. 특징의 중요도 확인하기



```python
# XGBClassifier 모델의 feature_importances_를 이용하여 중요도 plot




```

## Step5 모델 학습 결과 심화 분석하기


### 문제 14. Precision-Recall 커브 확인하기


```python
from sklearn.metrics import plot_precision_recall_curve
```


```python
# 두 모델의 Precision-Recall 커브를 한번에 그리기 (힌트: fig.gca()로 ax를 반환받아 사용)




```

### 문제 15. ROC 커브 확인하기


```python
from sklearn.metrics import plot_roc_curve
```


```python
# 두 모델의 ROC 커브를 한번에 그리기 (힌트: fig.gca()로 ax를 반환받아 사용)




```


```python
# pip을 통해 nbconvert 설치
pip install nbconvert
```


      File "<ipython-input-1-9d89de375132>", line 2
        pip install nbconvert
                  ^
    SyntaxError: invalid syntax


