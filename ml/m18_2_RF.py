#RandomForest

#RandomForest - 나무를 랜덤으로 모아가지고 구조를 가진다
#DecisionTree들이 모여있는 것이 RandomForest다
#트리구조의 모델들은 sklearn 에서는 성능이 최고 좋다! / tensorflow, keras보다
#갓 운전면허를 딴 사람이 페라리를 운전하는것과, 버스기사가 버스를 운전하는 것의 느낌!
#트리구조는 별도로 공부하자!

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #Randomforest 추가!
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터
cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target , train_size=0.8, random_state=42, shuffle=True)

#2. 모델 구성
# model = DecisionTreeClassifier(max_depth=4)

model = RandomForestClassifier(max_depth=4)
#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

acc = model.score(x_test, y_test)

print("acc : ", acc)
print(model.feature_importances_)

# acc :  0.9649122807017544
# [0.03521128 0.01816309 0.06044532 0.02861483 0.00772882 0.0180233
#  0.04531723 0.11542716 0.00392637 0.00299358 0.02038029 0.00348434
#  0.00334466 0.02719814 0.00189467 0.00432989 0.00319298 0.00370592
#  0.00315257 0.00344395 0.10777376 0.01655347 0.15278483 0.11657311
#  0.01165453 0.0141312  0.02939254 0.12258581 0.01228896 0.00628339]