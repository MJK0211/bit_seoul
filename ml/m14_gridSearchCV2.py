#grid_Search
#cancer데이터 - 2진분류
#모델 : RandomForestClassifier
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV #GridSearchCV 추가
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x_train = np.load('./data/npy/cancer_x_train.npy')
x_test = np.load('./data/npy/cancer_x_test.npy')
y_train = np.load('./data/npy/cancer_y_train.npy')
y_test = np.load('./data/npy/cancer_y_test.npy')

#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

kfold = KFold(n_splits=5, shuffle=True) #5개로 조각을 내고 섞어서 하겠다
# parameters =[{'n_estimators' : [100,200]}, 
#             # The number of trees in the forest.
#             {'max_depth' : [6, 8, 10, 12]},
#             # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
#             {'min_samples_leaf' : [3, 5, 7, 10]},
#             # The minimum number of samples required to be at a leaf node. 
#             # A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. 
#             # This may have the effect of smoothing the model, especially in regression.
#             {'min_samples_split' : [2, 3, 5, 10]},
#             # The minimum number of samples required to split an internal node
#             {'n_jobs' : [-1]}
#             # The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. 
#             # None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
# ]

parameters =[{'n_estimators' : [100, 200], 
              'max_depth' : [6, 8, 10, 12],
              'min_samples_leaf' : [3, 5, 7, 10],
              'min_samples_split' : [2, 3, 5, 10],
              'n_jobs' : [-1]}        
]

model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold) # SVC라는 모델을 GridSearchCV로 쓰겠다

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_) #파라미터중에 어떤 모델이 좋은지 가장 좋은 결과값을 리턴해준다, best_estimator = 가장좋은 평가자

y_pred = model.predict(x_test)

print("y_pred : ", y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("최종 정답률 : ", accuracy_score)

# 최적의 매개변수 :  RandomForestClassifier(max_depth=8) - 1번째 params
# y_pred :  [0 1 1 0 0 1 1 1 0 1 1 1 1 1 0 0 1 1 0 1 0 0 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1
#  1 1 1 0 0 1 0 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 0 0 1 0 0 0 1 0 0 0 1 0 1
#  0 1 1 1 1 0 1 0 0 1 0 0 0 1 1 1 0 0 1 1 1 1 0 0 0 1 0 1 0 1 1 0 1 0 1 1 1
#  1 0 0]
# 최종 정답률 :  0.9912280701754386

# 최적의 매개변수 :  RandomForestClassifier(max_depth=10, min_samples_leaf=3, min_samples_split=3,
#                        n_jobs=-1)
# y_pred :  [0 1 1 0 0 1 1 1 0 1 1 1 1 1 0 0 0 1 0 1 0 0 1 1 1 1 0 0 0 1 0 1 1 1 1 1 1
#  1 1 1 0 0 1 0 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 0 0 1 0 0 0 1 0 0 0 1 0 1
#  0 1 1 1 1 0 1 0 0 1 0 0 0 1 1 1 0 0 1 1 1 1 0 0 0 1 0 1 0 1 1 0 1 0 1 1 1
#  1 0 0]