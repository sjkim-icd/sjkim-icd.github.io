# SMOTE

지도학습에서 극도로 불균형한 레이블 값 분포로 인한 문제점을 해결하기 위해서는 적절한 학습 데이터를 확보하는 방안이 필요한데, 대표적으로 오버 샘플링(Oversampling)과 언더 샘플링(Undersampling) 방법이 있으며，오버 샘플 링 방식이 예측 성능상 더 유리한 경우가 많아주로 사용됩니다.

언더 샘플링은 많은 데이터 세트를 적은 데이터 세트 수준으로 감소시키는 방식입니다. 즉 정상 레이블을 가진 데이터가 10.000건, 이상 레이블을 가진 데이터가 100건이 있으면 정상 레이블 데이터를 100건으로 줄여 버리는 방식입니다. 이렇 게 정상 레이블 데이터를 이상 레이블 데이터 수준으로 줄여 버린 상태에서 학습을 수행하면 과도하게 정상 레이블로 학 습/예측하는 부작용을 개선할 수 있지만, 너무 많은 정상 레이블 데이터를 감소시키기 때문에 정상 레이블의 경우 오히려 제대로 된 학습을 수행할 수 없다는 단점이 있어 잘 적용하지 않는 방법입니다.

오버 샘플링은 이상 데이터와 같이 적은 데이터 세트를 증식하여 학습을 위한 충분한 데이터를 확보하는 방법입니다. 동일 한 데이터를 단순히 증식하는 방법은 고ᅡ적합(Overfitting)이 되기 때문에 의미가 없으므로 원본 데이터의 피처 값들을 아 주 약간만 변경하여 증식합니다. 대표적으로 SMOTE(Synthetic Minority Over-sampling Technique) 방법이 있습니다. SMOTE는 적은 데이터 세트에 있는 개별 데이터들의 K 최근접 이웃(K Nearest Neighbor)을 찾아서 이 데이터와 K개 이웃들의 차이를 일정 값으로 만들어서 기존 데이터와 약간 차이가 나는 새로운 데이터들을 생성하는 방식입니다.

SMOTE
- SMOTE의 동작 방식은 데이터의 개수가 적은 클래스의 표본을 가져온 뒤 임의의 값을 추가하여 새로운 샘플을 만들어 데이터에 추가하는 오버샘플링 방식이다.

-K-최근접 이웃으로 데이터 신규 증식 대상 설정하고 오버샘플링 : 소수 데이터 중 특정 벡터와 가장 가까운 이웃 사이의 차이를 계산하고, 이 차이에 0과 1사이의 난수를 곱함, 타겟 벡터를 추가

-smote 종류 
일반 smote

SMOTE-NC(nominal and continuous)

Borderline smote : 데이터 결정 경계 근처에서 데이터 합성

Smote svm : borderline smote의 변형

Adasyn: 데이터 밀도에 따라 합성 데이터 생성

[https://ichi.pro/ko/bulgyunhyeong-deiteoleul-obeo-saempeullinghagiwihan-5-gaji-smote-gibeob-202401874961077](https://ichi.pro/ko/bulgyunhyeong-deiteoleul-obeo-saempeullinghagiwihan-5-gaji-smote-gibeob-202401874961077)