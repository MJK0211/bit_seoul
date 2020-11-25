#XGB - XGBooster

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier #xgboost설치 후! XGBClassfier 추가!
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터
cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target , train_size=0.8, random_state=42, shuffle=True)


#2. 모델 구성
# model = DecisionTreeClassifier(max_depth=4) 제일 성능 안좋음
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(max_depth=4) 

#표본으로 잡는 것은 RF, GB, XGB이다

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

acc = model.score(x_test, y_test)

print("acc : ", acc)
print(model.feature_importances_)

# acc :  0.9649122807017544
# [6.5209707e-03 2.4647316e-02 5.1710028e-03 8.5555250e-03 3.9974600e-03
#  4.6066064e-03 2.6123745e-03 4.4031221e-01 3.4104299e-04 2.0658956e-03
#  1.2474300e-02 6.8988632e-03 1.7291201e-02 5.6377212e-03 3.1236352e-03
#  3.2256048e-03 2.7834374e-02 6.4999197e-04 7.1520107e-03 0.0000000e+00
#  7.2251976e-02 1.7675869e-02 8.9819685e-02 2.0090658e-02 1.0589287e-02
#  0.0000000e+00 1.2581742e-02 1.8795149e-01 5.9212563e-03 0.0000000e+00]

import matplotlib.pyplot as plt

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()