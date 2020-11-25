# 기준 xgboost
# 1. feature_importance 0인놈 제거
# 2. 하위 30% 제거
# 3. 디폴트와 성능 비교

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target , train_size=0.8, random_state=42, shuffle=True)

#2. 모델 구성
xgb = XGBClassifier() 

#3. 훈련
xgb.fit(x_train, y_train)

#1_1. 데이터
def find_feature_importances_iris(model): 
    f_import = model.feature_importances_
    find_index = sorted(range(len(f_import)), key=lambda i: f_import[i], reverse=True)[-1*round(len(f_import)*0.3):]
   
    return find_index

find_index = find_feature_importances_iris(xgb)

x_train = np.delete(x_train, find_index, axis=1)
x_test = np.delete(x_test, find_index, axis=1)

print(x_train.shape) # 4개에서 3개로 변환 - 상위 70프로의 데이터만 가져옴 
print(x_test.shape) # 4개에서 3개로 변환 - 상위 70프로의 데이터만 가져옴 

#2_1. 모델 구성
model = XGBClassifier() 

#3_1. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print("acc : ", acc)
print(model.feature_importances_)

# 결과값
# (120, 3)
# (30, 3)
# acc :  1.0
# [0.03839184 0.7532927  0.20831546]