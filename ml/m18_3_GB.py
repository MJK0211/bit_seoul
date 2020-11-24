#GB - Tree구조가 업그레이드 앙상블, 앙상블 업그레이드 된 것이 부스트!

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor # GrandientBoosting 추가
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터
cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target , train_size=0.8, random_state=42, shuffle=True)

#2. 모델 구성
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
model = GradientBoostingClassifier(max_depth=4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

acc = model.score(x_test, y_test)

print("acc : ", acc)
print(model.feature_importances_)

# acc :  0.956140350877193
# [2.28354741e-05 3.43537717e-02 9.30771061e-05 2.99380027e-05
#  5.17817865e-04 6.39788508e-04 7.02025968e-04 6.72562876e-01
#  1.30607174e-04 1.04055772e-04 4.46094678e-03 1.78208352e-03
#  6.51479088e-05 7.57443892e-03 2.25012949e-03 3.12375537e-04
#  7.51148989e-03 1.53588984e-02 6.27947925e-04 1.18314846e-02
#  5.76243352e-02 3.18939836e-02 5.49099097e-02 4.21949410e-03
#  8.72327181e-03 3.15643950e-04 1.92995686e-03 7.89660438e-02
#  4.81527811e-04 4.09650554e-06]