from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel #SelectFromModel 추가!

#############################################################################################################################################
#1. 데이터

nc = np.load('./project/data/npy/nc.npy', allow_pickle=True) #(2275,4)
am = np.load('./project/data/npy/america.npy', allow_pickle=True) #(2275,4)
weather = np.load('./project/data/npy/weather.npy', allow_pickle=True) #(2275,3)
snack_result = np.load('./project/data/npy/snack_result.npy', allow_pickle=True) #(2275,)
wage_result = np.load('./project/data/npy/wage_result.npy', allow_pickle=True) #(2275,)

nc = np.delete(nc, 0, axis=1)
am = np.delete(am, 0, axis=1)
weather = np.delete(weather, 0, axis=1)
snack_result = snack_result.reshape(2275,1)

#x데이터
nc_train, nc_test, am_train, am_test = train_test_split(nc, am , train_size=0.8, random_state=66, shuffle=True)
weather_train, weather_test, snack_train, snack_test = train_test_split(weather, snack_result , train_size=0.8, random_state=66, shuffle=True)

#y데이터
y_train, y_test = train_test_split(wage_result, train_size=0.8, random_state=66, shuffle=True)

train_all = np.concatenate((nc_train, am_train, weather_train, snack_train), axis=1)
test_all = np.concatenate((nc_test, am_test, weather_test, snack_test), axis=1)

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
model.fit(train_all, y_train)

#4. 평가, 예측
score = model.score(test_all, y_test)
print("score : ", score) 
model_best_estimator_ = model.best_estimator_        #최적의 모델 저장
print("최적의 매개변수 : ", model_best_estimator_) 

thresholds = model.best_estimator_.feature_importances_
print(model.best_estimator_.feature_importances_)

y_pred = model.predict(test_all[-1:])
from collections import deque #데쿠사용
find_index = deque()

for thresh in np.sort(thresholds): #데쿠사용 이유는 for문안에서 np.sort를 하지 않아도 thresh가 가장 작은 값부터 자동으로 들어가기때문에 n을 1부터 순차적으로 넣기 위함
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True) #최적의 모델 model.best_estimator로 선언해줌

    select_x_train = selection.transform(train_all)
    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)
    select_x_test = selection.transform(test_all)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)   
    print("Thresh=%.3f, n=%d, R2: %.2f%%, Index: %d" %(thresh, select_x_train.shape[1], score*100.0, int(np.where(thresholds==thresh)[0]))) #기존의 index위치를 저장하기 위함
    
    find_index.appendleft([select_x_train.shape[1], score*100.0, int(np.where(thresholds==thresh)[0])]) #index 추가 이후 데이터 왼쪽삽입

find_index = np.asarray(find_index)

score_max = np.array(find_index[0][0])

for i in range(len(find_index)) : 
    if find_index[i][1] > score_max :
       score_max = find_index[i][1]
       score_max_index = int(find_index[i][0])    #score의 최댓값을 찾고 그곳의 인덱스를 찾는다

print(score_max, score_max_index)                                    
find_index_list = (find_index[score_max_index:,2]).astype(int) #맥스값 이외의 데이터의 인덱스를 반환

#############################################################################################################################################

#1. 데이터
x_train = np.delete(train_all, find_index_list, axis=1)
x_test = np.delete(test_all, find_index_list, axis=1)

#2. 모델구성
model2 = model_best_estimator_

#3. 훈련
model2.fit(x_train, y_train)

#4. 평가, 예측 
score2 = model2.score(x_test, y_test)
print("score2 : ", score2)

y_pred = model.predict(x_test[-1:])
print("2021년 최저시급 : ", y_pred)

# score :  0.9982562535015219
# 최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=110, n_jobs=-1, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# [1.9582735e-02 1.1008872e-02 3.0371604e-02 1.6370450e-01 1.1665109e-03
#  1.3841151e-03 1.0679382e-02 1.4918424e-03 3.6854172e-04 6.7721523e-04
#  3.5514298e-04 7.5920951e-01]
# Thresh=0.000, n=12, R2: 99.80%, Index: 10
# Thresh=0.000, n=11, R2: 99.80%, Index: 8
# Thresh=0.001, n=10, R2: 99.82%, Index: 9
# Thresh=0.001, n=9, R2: 99.46%, Index: 4
# Thresh=0.001, n=8, R2: 99.47%, Index: 5
# Thresh=0.001, n=7, R2: 99.43%, Index: 7
# Thresh=0.011, n=6, R2: 99.32%, Index: 6
# Thresh=0.011, n=5, R2: 97.99%, Index: 1
# Thresh=0.020, n=4, R2: 97.89%, Index: 0
# Thresh=0.030, n=3, R2: 97.78%, Index: 2
# Thresh=0.164, n=2, R2: 97.93%, Index: 3
# Thresh=0.759, n=1, R2: 95.61%, Index: 11
# 99.82098751986184 10
# score2 :  0.9981063594033254
# 2021년 최저시급 :  [8630.503]