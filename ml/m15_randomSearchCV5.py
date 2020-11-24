#RandomizedSearchCV
#와인 - 분류
#모델 : RandomForestClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV #RandomizedSearchCV 추가
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_wine

#1. 데이터
x, y = load_wine(return_X_y=True) 

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train)
x_test_minmax = scaler.transform(x_test)

#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

kfold = KFold(n_splits=5, shuffle=True) #5개로 조각을 내고 섞어서 하겠다

parameters =[{'n_estimators' : [100, 200], 
              'max_depth' : [6, 8, 10, 12],
              'min_samples_leaf' : [3, 5, 7, 10],
              'min_samples_split' : [2, 3, 5, 10],
              'n_jobs' : [-1]}        
]

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold) 

#3. 훈련
model.fit(x_train_minmax, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test_minmax)

print("y_pred : ", y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("최종 정답률 : ", accuracy_score)

# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=5, min_samples_split=3,
#                        n_jobs=-1)
# y_pred :  [2 1 1 0 1 1 2 0 0 1 2 0 1 1 1 2 2 0 1 2 1 0 0 0 0 0 1 1 0 1 1 0 2 0 1 0]
# 최종 정답률 :  1.0

