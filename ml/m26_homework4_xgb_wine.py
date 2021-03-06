# pipe라인 까지 구성할 것
# 와인은 csv로 할 것

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
#1. 데이터

wine = pd.read_csv('./data/csv/winequality-white.csv',
                 header = 0,
                 index_col=None,
                 sep=';')

y = wine['quality']
x = wine.drop('quality', axis=1)

x = x.values
y = y.values
            
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size=0.8, random_state=66, shuffle=True)
# pipe = make_pipeline(MinMaxScaler(), SVC()) #Scaler를 사용할 때, crossvalidation 과적합을 피하기위해 묶어줌
pipe = Pipeline([("scaler", MinMaxScaler()), ('XGB', XGBClassifier())])

#2. 모델 구성

kfold = KFold(n_splits=5, shuffle=True) #5개로 조각을 내고 섞어서 하겠다
parameters = [{"XGB__n_estimators":[100, 200, 300], "XGB__learning_rate":[0.1, 0.3, 0.001, 0.01],
               "XGB__max_depth":[4, 5, 6]},
              {"XGB__n_estimators":[90, 100, 110], "XGB__learning_rate":[0.1, 0.001, 0.01],
               "XGB__max_depth":[4, 5, 6], "XGB__colsample_bytree":[0.6, 0.9, 1]}, 
              {"XGB__n_estimators":[90, 110], "XGB__learning_rate":[0.1, 0.001, 0.05],
               "XGB__max_depth":[4, 5, 6], "XGB__colsample_bytree":[0.6, 0.9, 1], "XGB__colsample_bylevel":[0.6, 0.7, 0.9]}
]

#learning_rate가 가장 중요하다!

model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=True) # SVC라는 모델을 GridSearchCV로 쓰겠다

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print("acc : ", acc) 
print("최적의 매개변수 : ", model.best_estimator_) 

# acc :  0.6448979591836734
# 최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('XGB',
#                  XGBClassifier(base_score=0.5, booster='gbtree',
#                                colsample_bylevel=0.6, colsample_bynode=1,
#                                colsample_bytree=0.9, gamma=0, gpu_id=-1,
#                                importance_type='gain',
#                                interaction_constraints='', learning_rate=0.05,
#                                max_delta_step=0, max_depth=6,
#                                min_child_weight=1, missing=nan,
#                                monotone_constraints='()', n_estimators=90,
#                                n_jobs=0, num_parallel_tree=1,
#                                objective='multi:softprob', random_state=0,
#                                reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
#                                subsample=1, tree_method='exact',
#                                validate_parameters=1, verbosity=None))])