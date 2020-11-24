#decisionTree

#RandomForest - 나무를 랜덤으로 모아가지고 구조를 가진다
#DecisionTree들이 모여있는 것이 RandomForest다
#트리구조의 모델들은 sklearn 에서는 성능이 최고 좋다! / tensorflow, keras보다
#갓 운전면허를 딴 사람이 페라리를 운전하는것과, 버스기사가 버스를 운전하는 것의 느낌!
#트리구조는 별도로 공부하자!

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor #decisiontree 분류, 회귀 추가!
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터
cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target , train_size=0.8, random_state=42, shuffle=True)

#2. 모델 구성
model = DecisionTreeClassifier(max_depth=4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

acc = model.score(x_test, y_test)

print("acc : ", acc)
print(model.feature_importances_)

# acc :  0.9473684210526315
# [0.         0.05959094 0.         0.         0.         0.
#  0.         0.70458252 0.         0.         0.         0.00639525
#  0.         0.01221069 0.         0.         0.         0.0162341
#  0.         0.0189077  0.05329492 0.         0.05247428 0.
#  0.00940897 0.         0.         0.06690062 0.         0.        ]