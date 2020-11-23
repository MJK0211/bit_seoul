import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_iris

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor #classfier : 분류, regressor : 회귀, 예외) logistic_regressor만 분류이다!
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #앙상블에 대표적인 모델 RandomForest - feature importance! 매우중요!
from sklearn.model_selection import train_test_split

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8) #shuffle로 섞을 경우, random난수로 섞는다

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train)
x_test_minmax = scaler.transform(x_test)

#2. 모델 
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
model = RandomForestRegressor()

#3. 훈련

model.fit(x_train_minmax, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test_minmax)
print(y_test[:10], "의 예측결과 : \n", y_pred[:10])
score = model.score(x_test_minmax, y_test)
print("score : ", score)
acc_score = accuracy_score(y_test, y_pred)
# acc_score = accuracy_score(y_test, y_pred.round())
# acc_score = r2_score(y_test, y_pred)
print("acc_score : ", acc_score)

#분류모델에서만 score = acc_score
#회귀모델에서는 score = r2_score

# LinearSVC()
# [1 1 1 0 1 1 0 0 0 2] 의 예측결과 :
#  [1 1 1 0 1 1 0 0 0 2]
# score :  0.9666666666666667
# acc_score :  0.9666666666666667

# SVC()
# [1 1 1 0 1 1 0 0 0 2] 의 예측결과 :
#  [1 1 1 0 1 1 0 0 0 2]
# score :  1.0
# acc_score :  1.0

# KNeighborsClassifier()
# [1 1 1 0 1 1 0 0 0 2] 의 예측결과 :
#  [1 1 1 0 1 1 0 0 0 2]
# score :  1.0
# acc_score :  1.0

# KNeighborsRegressor() 
# [1 1 1 0 1 1 0 0 0 2] 의 예측결과 :
#  [1.  1.  1.2 0.  1.2 1.  0.  0.  0.  1.6]
# score :  0.961744966442953
# acc_score :  0.961744966442953

# RandomForestClassfier()
# [1 1 1 0 1 1 0 0 0 2] 의 예측결과 :
#  [1 1 1 0 1 1 0 0 0 2]
# score :  0.9666666666666667
# acc_score :  0.9666666666666667

# # RandomForestRegressor()
# [1 1 1 0 1 1 0 0 0 2] 의 예측결과 :
#  [1.   1.14 1.01 0.   1.   1.   0.   0.   0.   1.35]
# score :  0.9591677852348993
# acc_score :  0.9591677852348993