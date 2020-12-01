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

############################################################################# 최적의 파라미터와, 최고의 값을 위한 데이터 전처리과정

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

model = GridSearchCV(XGBRegressor(n_jobs=-1), parameters, cv=kfold, verbose=True)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print("score : ", score) 
model_best_estimator_ = model.best_estimator_             #최적의 모델 저장
print("최적의 매개변수 : ", model_best_estimator_) 

# print(model.best_estimator_.feature_importances_)

# score :  0.9475222198375111
# 최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.05, max_delta_step=0, max_depth=5,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=110, n_jobs=-1, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)

thresholds = model.best_estimator_.feature_importances_ #threshold - 피처임포턴스 값 저장

from collections import deque #데쿠사용
find_index = deque()

# print(find_index)
for thresh in np.sort(thresholds): #데쿠사용 이유는 for문안에서 np.sort를 하지 않아도 thresh가 가장 작은 값부터 자동으로 들어가기때문에 n을 1부터 순차적으로 넣기 위함
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True) #최적의 모델 model.best_estimator로 선언해줌

    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)   
    print("Thresh=%.3f, n=%d, R2: %.2f%%, Index: %d" %(thresh, select_x_train.shape[1], score*100.0, int(np.where(thresholds==thresh)[0]))) #기존의 index위치를 저장하기 위함
    
    find_index.appendleft([select_x_train.shape[1], score*100.0, int(np.where(thresholds==thresh)[0])]) #index 추가 이후 데이터 왼쪽삽입

# Thresh=0.002, n=13, R2: 92.21%, Index: 1
# Thresh=0.003, n=12, R2: 91.96%, Index: 3
# Thresh=0.012, n=11, R2: 92.03%, Index: 11
# Thresh=0.017, n=10, R2: 92.19%, Index: 6
# Thresh=0.018, n=9, R2: 93.08%, Index: 8
# Thresh=0.025, n=8, R2: 93.52%, Index: 2
# Thresh=0.026, n=7, R2: 92.86%, Index: 9
# Thresh=0.035, n=6, R2: 91.79%, Index: 0
# Thresh=0.048, n=5, R2: 91.74%, Index: 7
# Thresh=0.086, n=4, R2: 91.47%, Index: 4
# Thresh=0.108, n=3, R2: 78.35%, Index: 10
# Thresh=0.266, n=2, R2: 69.41%, Index: 5
# Thresh=0.354, n=1, R2: 44.98%, Index: 12

print(thresholds[5:])
'''
print(type(find_index)) #<class 'collections.deque'>
find_index = np.asarray(find_index) #데쿠 타입 데이터를 numpy로 변환
print(find_index)

# [[ 1.         44.98447068 12.        ]
#  [ 2.         69.40989373  5.        ]
#  [ 3.         78.34727178 10.        ]
#  [ 4.         91.46624254  4.        ]
#  [ 5.         91.7356489   7.        ]
#  [ 6.         91.79357158  0.        ]
#  [ 7.         92.8571721   9.        ]
#  [ 8.         93.51663442  2.        ]
#  [ 9.         93.07724256  8.        ]
#  [10.         92.19134406  6.        ]
#  [11.         92.03131446 11.        ]
#  [12.         91.95644107  3.        ]
#  [13.         92.21188545  1.        ]]

score_max = np.array(find_index[0][0])

for i in range(len(find_index)) : 
    if find_index[i][1] > score_max :
       score_max = find_index[i][1]
       score_max_index = int(find_index[i][0])    #score의 최댓값을 찾고 그곳의 인덱스를 찾는다

print(score_max, score_max_index)                                    
find_index_list = (find_index[score_max_index:,2]).astype(int) #맥스값 이외의 데이터의 인덱스를 반환
print(find_index_list)

# 93.5166344243661 8
# [ 8  6 11  3  1]
###############################################################################################################################

#1. 데이터
print(x_train)
# [[4.12380e-01 0.00000e+00 6.20000e+00 ... 1.74000e+01 3.72080e+02
#   6.36000e+00]
#  [1.35540e-01 1.25000e+01 6.07000e+00 ... 1.89000e+01 3.96900e+02
#   1.30900e+01]
#  [2.53870e-01 0.00000e+00 6.91000e+00 ... 1.79000e+01 3.96900e+02
#   3.08100e+01]
#  ...
#  [1.44760e-01 0.00000e+00 1.00100e+01 ... 1.78000e+01 3.91500e+02
#   1.36100e+01]
#  [3.18270e-01 0.00000e+00 9.90000e+00 ... 1.84000e+01 3.90700e+02
#   1.83300e+01]
#  [1.25179e+00 0.00000e+00 8.14000e+00 ... 2.10000e+01 3.76570e+02
#   2.10200e+01]]

x_train = np.delete(x_train, find_index_list, axis=1)
x_test = np.delete(x_test, find_index_list, axis=1)

print(x_train.shape) # (404, 8)

#2. 모델구성
model2 = model_best_estimator_

#3. 훈련
model2.fit(x_train, y_train)

#4. 평가, 예측 
score2 = model2.score(x_test, y_test)
print("score2 : ", score2)

# 결과값
# score2 :  0.9321348631150079
'''