# 실습
# 1. 상단 모델에 그리드서치 또는 랜덤서치 적용
# 최적의 R2값과 Feature_importance 구할것

# 2. 위 쓰레드 값으로 SelectFromModel을 구해서
# 최적의 피처 갯수를 구할 것

# 3. 위 피어 갯수로 데이터(피처)를 수정(삭제)해서
# 그리드서치 또는 랜덤서치 적용해서
# 최적의 R2값을 구할 것

# 1번값과 2번값을 비교해 볼것!

import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel #SelectFromModel 추가!
from sklearn.metrics import r2_score

#1. 데이터
boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target , train_size=0.8, random_state=66, shuffle=True)

#2. 모델 구성

kfold = KFold(n_splits=5, shuffle=True) #5개로 조각을 내고 섞어서 하겠다
parameters = [{"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
               "max_depth":[4, 5, 6]},
              {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
               "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1]}, 
              {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.05],
               "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}
]

model = RandomizedSearchCV(XGBRegressor(n_jobs=-1), parameters, cv=kfold, verbose=True)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print("score : ", score) 
print("최적의 매개변수 : ", model.best_estimator_) 

# print(model.best_estimator_.feature_importances_)

# score :  0.9428382288231046
# 최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
#              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
#              learning_rate=0.1, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=110, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)

# [0.02943155 0.00274769 0.01431982 0.00519528 0.07712355 0.30340967
#  0.01534717 0.04145774 0.01057774 0.01834476 0.21045421 0.01070204
#  0.2608888 ]

thresholds = model.best_estimator_.feature_importances_
print(thresholds)


print(x_train.shape)

# for thresh in np.sort(thresholds):
      
#     score_max = 0
#     selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True)

#     select_x_train = selection.transform(x_train)
#     selection_model = XGBRegressor(n_jobs=-1)
#     selection_model.fit(select_x_train, y_train)
#     select_x_test = selection.transform(x_test)
#     y_predict = selection_model.predict(select_x_test)

#     score = r2_score(y_test, y_predict)
#     print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
#     if score > score_max:
#         score_max = score
#         find_index.append(thresh)

score_max=[]
# print(find_index)
for thresh in np.sort(thresholds):
    # index_thresh = int(np.where(thresholds==thresh)[0])
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)   
    print("Thresh=%.3f, n=%d, R2: %.2f%%, Index: %d" %(thresh, select_x_train.shape[1], score*100.0, int(np.where(thresholds==thresh)[0])))
    
    find_index = np.insert([select_x_train.shape[1], score*100.0, int(np.where(thresholds==thresh)[0])], 0, 0)

print(find_index)

# print(len(find_index))
# print(find_index)
# x_max = np.array(find_index[0][1])
# x_max=[]
# for i in range(len(find_index)):
#      if find_index[i][1] > x_max :
#         x_max = find_index[i][1]
#         print(i)    

# find_index = np.array(find_index)

# for i in range(len(x)) : 
#     if x[i][2] > x_max :
#        x_max = x[i][2]     

# find_index = sorted(range(len(find_index)), key=lambda i: find_index[i], reverse=True)[-1:]

# Thresh=0.001, n=13, R2: 92.21%, Index: 1
# Thresh=0.003, n=12, R2: 91.96%, Index: 3
# Thresh=0.011, n=11, R2: 92.03%, Index: 8
# Thresh=0.014, n=10, R2: 93.07%, Index: 11
# Thresh=0.018, n=9, R2: 93.03%, Index: 6
# Thresh=0.018, n=8, R2: 93.52%, Index: 2
# Thresh=0.019, n=7, R2: 92.86%, Index: 9
# Thresh=0.035, n=6, R2: 91.79%, Index: 0
# Thresh=0.041, n=5, R2: 91.74%, Index: 7
# Thresh=0.074, n=4, R2: 91.47%, Index: 4
# Thresh=0.147, n=3, R2: 78.35%, Index: 10
# Thresh=0.285, n=2, R2: 69.41%, Index: 12
# Thresh=0.333, n=1, R2: 41.31%, Index: 5
