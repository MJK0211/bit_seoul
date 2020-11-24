#grid_Search
#보스턴 - 회귀
#모델 : RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV #GridSearchCV 추가
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x_train = np.load('./data/npy/boston_x_train.npy')
x_test = np.load('./data/npy/boston_x_test.npy')
y_train = np.load('./data/npy/boston_y_train.npy')
y_test = np.load('./data/npy/boston_y_test.npy')

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
print("최적의 매개변수 : ", model.best_estimator_) #파라미터중에 어떤 모델이 좋은지 가장 좋은 결과값을 리턴해준다, best_estimator = 가장좋은 평가자

y_pred = model.predict(x_test)

print("y_pred : ", y_pred)
r2 = r2_score(y_test, y_pred)
print("최종 정답률 : ", r2)

# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, min_samples_leaf=3, min_samples_split=3,
#                       n_jobs=-1)
# y_pred :  [20.50010904 27.01398606 22.05927904 23.37073934 18.52636855 14.94556755
#  31.44701003 19.55037153 19.5257634  20.60399795 14.77259517 20.74432864
#  21.38682475 24.07922343 20.62176822 21.4467115  15.39330793 22.35677438
#  16.20117539 10.88872253 20.4140492  19.59986341 14.92357672 11.42988451
#  37.55016984 23.05658765 20.5422071  19.78075222 15.99549744  8.32029954
#  43.57187976 25.85626358 31.33945022 36.33802653  9.38315726 26.38927079
#  22.14524742 20.16368993 23.68320911 19.83361793 34.32848651 17.292351
#  24.97873694 20.63334647  8.60432627 19.37775305 18.83987673 36.99149017
#  20.85652093 16.09754126 20.25348902 21.71745313 19.69276809 20.81580079
#  21.38407594 26.49687226 25.21312327 14.17759275 23.60561668 29.29706225
#  19.26366402 27.90508716 14.18497831 20.94185761 21.14675591 21.0871483
#  13.37939603 21.27205446 21.46919908 29.9733688  32.1546518  22.76559972
#  16.51739687 26.08121864 43.06576259  9.78656504 34.0983394  21.37670231
#  39.14243896 24.7211381  21.81384707 16.85689477 20.21476101 11.74597277
#  29.44294946 24.12926809 21.96649813 33.55185375 25.83124224 20.19950029
#   8.8945736  22.19073183 20.62995501 19.39899059 16.2401493  15.06106016
#  23.66160562 28.2312431  17.04603654 21.0091162  29.45576659 35.65817126]
# 최종 정답률 :  0.8709204986372716