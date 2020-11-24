#RandomizedSearchCV
#당뇨병 - 회귀
#모델 : RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV #RandomizedSearchCV 추가
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x_train = np.load('./data/npy/diabetes_x_train.npy')
x_test = np.load('./data/npy/diabetes_x_test.npy')
y_train = np.load('./data/npy/diabetes_y_train.npy')
y_test = np.load('./data/npy/diabetes_y_test.npy')

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

# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_leaf=7, min_samples_split=5,
#                       n_estimators=200, n_jobs=-1)
# y_pred :  [114.3253701   78.38343931 131.4293081  125.1543327  242.84733669
#  166.438264   260.35952254 126.53894784 141.92462945  87.94114998
#  175.0307144  174.35018868  77.93285297 105.20656325 121.41839218
#  181.69775171 174.6340721  135.23295224  91.29568922  93.87928869
#   80.04133695 152.14719481  89.26934819 243.8997225   93.61377338
#  106.79661444 211.88962216 224.25926288 262.28531844 139.76080741
#  100.14386647 199.67964828 117.78013176  82.00103141 151.2999318
#  148.67958397 198.36667481 214.13579336 214.21355078 152.61008505
#  230.16288599 138.94229063 196.90611715 219.26506991 185.84541327
#  125.24245294 216.33063668 170.4273483  238.06927899 165.05829965
#   76.60810549  78.48646222 164.45230987 108.58837147 216.94277396
#  123.63265501 199.8934827   95.29242968 175.09723197  79.96123347
#   82.95778468 186.8356366   94.65447695 194.73493695 159.68438253
#  112.55355097 105.98759824 133.89361185  74.8142631  169.60248987
#   89.50600429 164.94201629 154.65933418 103.24470181 158.70472618
#  133.37393102 174.79581598 119.86739465 182.36142917 204.13612981
#  110.61130739 189.90788914 171.14700488 197.72627796 164.42650351
#  269.51076701 104.46851285 142.45794778 119.60810858]
# 최종 정답률 :  0.4661948097352815