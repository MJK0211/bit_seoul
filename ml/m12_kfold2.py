import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score #KFold 추가! - 모델에 관여됨, cross_val_score 추가!

#1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1] #(150, 4) (150,)

# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

kfold = KFold(n_splits=5, shuffle=True) #5개로 조각을 내고 섞어서 하겠다

model1 = LinearSVC()
scores1 = cross_val_score(model1, x_train, y_train, cv=kfold)
# LinearSVC의 정답률 :  [0.83333333 0.70833333 0.625      0.95833333 0.83333333]

model2 = SVC()
scores2 = cross_val_score(model2, x_train, y_train, cv=kfold) 
print('SVC 정답률 : ', scores2)
# SVC 정답률 :  [0.95833333 0.95833333 1.         0.91666667 1.        ]

model3 = KNeighborsClassifier()
scores3 = cross_val_score(model3, x_train, y_train, cv=kfold) 
print('KNeighborsClassifier 정답률 : ', scores3)
# KNeighborsClassifier 정답률 :  [1. 1. 1. 1. 1.]

model4 = KNeighborsRegressor()
scores4 = cross_val_score(model4, x_train, y_train, cv=kfold) 
print('KNeighborsRegressor 정답률 : ', scores4)
# KNeighborsRegressor 정답률 :  [0.9872     1.         0.94729412 1.         0.95897436]

model5 = RandomForestClassifier()
scores5 = cross_val_score(model5, x_train, y_train, cv=kfold) 
print('RandomForestClassifier 정답률 : ', scores5)
# RandomForestClassifier 정답률 :  [1. 1. 1. 1. 1.]

model6 = RandomForestRegressor()
scores6 = cross_val_score(model6, x_train, y_train, cv=kfold) 
print('RandomForestRegressor 정답률 : ', scores6)
# RandomForestRegressor 정답률 :  [0.9993271  0.99993793 0.95089153 0.99044832 0.99782857]