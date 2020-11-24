#pipeline

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline, make_pipeline #Pipeline, make_pipeline 추가 - Scaler를 엮어보자
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler #Scaler 4개 추가

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 


#1. 데이터
x_train = np.load('./data/npy/boston_x_train.npy')
x_test = np.load('./data/npy/boston_x_test.npy')
y_train = np.load('./data/npy/boston_y_train.npy')
y_test = np.load('./data/npy/boston_y_test.npy')

# pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor()) #Scaler를 사용할 때, crossvalidation 과적합을 피하기위해 묶어줌
pipe = Pipeline([("scaler", MinMaxScaler()), ('RFR', RandomForestRegressor())])

#2. 모델 구성
kfold = KFold(n_splits=5, shuffle=True) 

# parameters =[{'n_estimators' : [100, 200], 
#               'max_depth' : [6, 8, 10, 12],
#               'min_samples_leaf' : [3, 5, 7, 10],
#               'min_samples_split' : [2, 3, 5, 10],
#               'n_jobs' : [-1]}        
# ]

parameters =[{'RFR__n_estimators' : [100, 200], 
              'RFR__max_depth' : [6, 8, 10, 12],
              'RFR__min_samples_leaf' : [3, 5, 7, 10],
              'RFR__min_samples_split' : [2, 3, 5, 10],
              'RFR__n_jobs' : [-1]}        
]

model = RandomizedSearchCV(pipe, parameters, cv=kfold)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('acc : ', model.score(x_test, y_test))
print("최적의 매개변수 : ", model.best_estimator_) 

# acc :  0.8690955347237634
# 최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('RFR',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=3,
#                                        min_samples_split=3, n_estimators=200,
#                                        n_jobs=-1))])