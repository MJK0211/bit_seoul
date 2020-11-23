#RF로 모델 만들것

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor #classfier : 분류, regressor : 회귀, 예외) logistic_regressor만 분류이다!
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #앙상블에 대표적인 모델 RandomForest - feature importance! 매우중요!
from sklearn.model_selection import train_test_split

#1. 데이터
# df = pd.read_csv('./data/csv/winequality-white.csv',
#                   index_col=None,
#                   header=0,
#                   encoding='cp949',
#                   sep=';') 
# print(df.shape) #(4898, 0)
# wine = df.values
# np.save('./data/npy/winequality-white.npy', arr=wine)

wine = np.load('./data/npy/winequality-white.npy', allow_pickle=True).astype('float32')
wine_x = wine[:, :11]
wine_y = wine[:, 11:]

x_train, x_test, y_train, y_test = train_test_split(wine_x, wine_y, random_state=66, shuffle=True, train_size=0.8) #shuffle로 섞을 경우, random난수로 섞는다

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train)
x_test_minmax = scaler.transform(x_test)

#2. 모델 
model = RandomForestClassifier(n_estimators=100)

#3. 훈련
model.fit(x_train_minmax, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test_minmax)
print(y_test[20:30].reshape(10,), "의 예측결과 : \n", y_pred[20:30])
score = model.score(x_test_minmax, y_test)
print("score : ", score)
acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", acc_score)

# RandomForestClassfier()
# [6. 4. 6. 5. 5. 6. 6. 7. 6. 6.] 의 예측결과 :
#  [6. 6. 5. 6. 5. 6. 7. 6. 6. 6.]
# score :  0.713265306122449
# acc_score :  0.713265306122449
