#SelectFromModel

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel #SelectFromModel 추가!
from sklearn.metrics import r2_score

#1. 데이터
boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target , train_size=0.8, random_state=42, shuffle=True)

#2. 모델 구성

model = XGBRegressor(n_jobs=1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측 
score = model.score(x_test, y_test)
print("R2 : ", score)

thresholds = np.sort(model.feature_importances_)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))


# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)

#     select_x_train = selection.transform(x_train)
#     selection_model = XGBRegressor(n_jobs=-1)
#     selection_model.fit(select_x_train, y_train)

#     select_x_test = selection.transform(x_test)
#     # y_predict = selection_model.predict(select_x_test)

#     score2 = selection_model.score(select_x_test, y_test)
#     print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score2*100.0))
# 위의 for문과 같은 결과이다

# 결과값
# R2 :  0.9105388059813951
# Thresh=0.002, n=13, R2: 91.05%
# Thresh=0.003, n=12, R2: 90.79%
# Thresh=0.006, n=11, R2: 91.07%
# Thresh=0.009, n=10, R2: 89.04%
# Thresh=0.009, n=9, R2: 90.97%
# Thresh=0.015, n=8, R2: 90.14%
# Thresh=0.021, n=7, R2: 91.35%
# Thresh=0.039, n=6, R2: 89.23%
# Thresh=0.042, n=5, R2: 88.66%
# Thresh=0.048, n=4, R2: 85.15%
# Thresh=0.055, n=3, R2: 84.45%
# Thresh=0.370, n=2, R2: 56.31%
# Thresh=0.380, n=1, R2: 54.00%