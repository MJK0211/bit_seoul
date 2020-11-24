#grid_Search
#당뇨병 - 회귀
#모델 : RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV #GridSearchCV 추가
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x_train = np.load('./data/npy/diabetes_x_train.npy')
x_test = np.load('./data/npy/diabetes_x_test.npy')
y_train = np.load('./data/npy/diabetes_y_train.npy')
y_test = np.load('./data/npy/diabetes_y_test.npy')

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

model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold) # SVC라는 모델을 GridSearchCV로 쓰겠다

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_) 

y_pred = model.predict(x_test)

print("y_pred : ", y_pred)
r2 = r2_score(y_test, y_pred)
print("최종 정답률 : ", r2)

# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, min_samples_leaf=10, n_jobs=-1)
# y_pred :  [115.87098666  81.00142844 130.08248988 123.10208569 245.82302397
#  163.9000301  257.12445554 127.05072769 144.45889692  85.65828478
#  172.48641316 169.82354839  82.71115457 107.15377347 124.953999
#  182.12053982 168.91855502 137.0233234   91.66571853 103.26173322
#   81.97591751 152.93777173  90.90059747 240.69987408  94.95889208
#   99.67312569 206.93328352 233.9685773  259.36166346 137.69735361
#   97.34628589 202.10983273 119.21246505  85.35651214 155.15758294
#  154.80578525 193.55371883 220.34773024 212.21916307 146.03627106
#  224.26173321 145.70126787 193.18158246 224.37097312 192.83823595
#  118.24952856 208.11713844 171.90493563 231.27576886 165.69684101
#   80.40609961  76.67338526 155.71860298 113.26811959 212.14109982
#  122.88173672 193.89057343  98.19947501 177.23443793  83.61444151
#   81.78203724 188.05540375  89.79440029 190.53205815 160.95034645
#  113.00298885 105.15086628 132.81240015  78.00900484 173.20357854
#   90.67208463 171.99589673 152.32623145 106.18465666 158.72813619
#  133.45286707 187.10018647 115.4524631  176.34132364 200.05980624
#  114.80737482 192.22126621 167.85076789 199.30722742 167.41947915
#  263.24840739 105.706141   151.88604035 120.14795314]
# 최종 정답률 :  0.4658466273541628