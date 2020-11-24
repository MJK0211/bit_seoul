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
wine = pd.read_csv('./data/csv/winequality-white.csv',
                 header = 0,
                 index_col=None,
                 sep=';')

y = wine['quality']
x = wine.drop('quality', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66, shuffle=True)

# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier()) #Scaler를 사용할 때, crossvalidation 과적합을 피하기위해 묶어줌
pipe = Pipeline([("scaler", MinMaxScaler()), ('RFC', RandomForestClassifier())])

#2. 모델 구성
kfold = KFold(n_splits=5, shuffle=True) 

# parameters =[{'n_estimators' : [100, 200], 
#               'max_depth' : [6, 8, 10, 12],
#               'min_samples_leaf' : [3, 5, 7, 10],
#               'min_samples_split' : [2, 3, 5, 10],
#               'n_jobs' : [-1]}        
# ]

parameters =[{'RFC__n_estimators' : [100, 200], 
              'RFC__max_depth' : [6, 8, 10, 12],
              'RFC__min_samples_leaf' : [3, 5, 7, 10],
              'RFC__min_samples_split' : [2, 3, 5, 10],
              'RFC__n_jobs' : [-1]}        
]

model = RandomizedSearchCV(pipe, parameters, cv=kfold)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('acc : ', model.score(x_test, y_test)) 
print("최적의 매개변수 : ", model.best_estimator_) 

# acc :  0.6612244897959184
# 최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('RFC',
#                  RandomForestClassifier(max_depth=12, min_samples_leaf=5,
#                                         min_samples_split=10, n_jobs=-1))])