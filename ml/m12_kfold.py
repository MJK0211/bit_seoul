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

model = SVC()

scores = cross_val_score(model, x_train, y_train, cv=kfold) #검증된 scores를 모델과 엮어줌

print('scores의 정답률 : ', scores)
# scores의 정답률 :  [1.         0.95833333 0.91666667 0.95833333 0.95833333]
