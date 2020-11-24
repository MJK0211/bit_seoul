#RandomizedSearchCV
#보스턴 - 회귀
#모델 : RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV #RandomizedSearchCV 추가
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x_train = np.load('./data/npy/boston_x_train.npy')
x_test = np.load('./data/npy/boston_x_test.npy')
y_train = np.load('./data/npy/boston_y_train.npy')
y_test = np.load('./data/npy/boston_y_test.npy')

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

model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold) 

#3. 훈련
model.fit(x_train_minmax, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test_minmax)

print("y_pred : ", y_pred)
r2 = r2_score(y_test, y_pred)
print("최종 정답률 : ", r2)

# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, min_samples_leaf=3, min_samples_split=10,
#                       n_estimators=200, n_jobs=-1)
# y_pred :  [21.35168638 27.09839452 22.25161773 23.43725971 18.64943408 15.27978252
#  31.35700681 19.83548291 19.3829626  20.82745162 15.17734137 20.63478659
#  21.38021545 23.69569534 20.00939838 21.94088854 15.80719193 22.25193801
#  16.3180353  10.87798178 20.15402368 19.84293157 15.05276103 11.49225719
#  36.65471327 23.18077644 20.74111954 20.40845826 16.21104234  8.60886793
#  43.94515842 25.60815223 31.13862879 35.70373648  9.47720149 26.56414622
#  22.11885993 20.20383607 23.28585541 20.01928471 33.4535529  17.47512557
#  24.97423028 20.85800311  8.70316101 19.77371048 18.92081385 36.43022377
#  20.95163489 15.97889544 20.58951912 21.08499854 19.56895584 20.79391647
#  21.6646568  26.80962829 25.69794191 13.91215155 23.59064227 29.61047736
#  19.53374629 28.60887262 14.02709793 21.06824566 20.83159177 21.21531503
#  12.8003017  21.01573217 21.70976228 30.14284276 32.80027182 22.48603812
#  17.22121043 25.72135382 43.03638362  9.80872987 33.85875583 21.23411884
#  39.18291207 24.48137955 21.74762366 16.91746521 20.19633542 11.71425428
#  29.7777787  24.18454465 21.80630747 33.6402883  26.79302121 20.48586321
#   9.42512984 21.94732654 20.28576073 19.21422846 16.35020931 15.47204531
#  23.63492796 28.47601602 16.88819543 20.81859502 29.15960441 35.24198471]
# 최종 정답률 :  0.8741214864082905