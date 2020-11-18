#load-boston - DNN

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.datasets import load_boston #load_boston 주택가격 데이터셋 추가

#1. 데이터
# Attribute Information (in order):
#     0    - CRIM     자치시 별 1인당 범죄율
#     1    - ZN       25,000 평방피트를 초과하는 거주지역의 비율
#     2    - INDUS    비소매상업지역이 점유하고 있는 토지의 면적 비율
#     3    - CHAS     찰스강의 경계에 위치 유무 (1:찰스강 경계에 존재, 0:찰스강 경계에 존재X)
#     4    - NOX      10ppm 당 일산화질소 농도
#     5    - RM       주택 당 평균 방의 갯수
#     6    - AGE      1940년 이전에 건축된 주택의 비율
#     7    - DIS      직업 센터의 거리
#     8    - RAD      방사형 고속도로까지의 거리
#     9    - TAX      $10,000 당 재산세율
#    10    - PTRATIO  학생/교사 비율
#    11    - B        인구 중 흑인 비율
#    12    - LSTAT    인구 중 하위 계층 비율
#    13    - MEDV     소유주가 거주하는 주택의 가치 (단위 : $ 1000)

dataset = load_boston()
x = dataset.data #(506,13), 인덱스 0~12의 값
y = dataset.target #(506,), 인덱스 13의 값 - 소유주가 거주하는 주택의 가치 (단위 : $ 1000)
#x의 데이터로 본인의 주택의 가치를 평가하는 데이터 셋이다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.8) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler #데이터 전처리 StandardScaler 추가
scaler = StandardScaler()
scaler.fit(x_train)
x_train_standard = scaler.transform(x_train) 
x_test_standard = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(13,)))
model.add(Dropout(0.2))
model.add(Dense(180, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(110, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='min') 

model.fit(x_train_standard, y_train, epochs=500, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test_standard, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test_standard)
print("y_test : ", y_test)
print("y_predict : \n", np.round(y_predict.reshape(102,),1))

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
 
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결과값
# loss :  16.358922958374023
# y_test :  [21.1 22.6 13.8 27.1 16.2 13.1  7.  18.6 21.  24.4 10.5 23.7 33.2 23.3
#  19.  26.7 22.  29.1 20.  27.5 19.6 37.2 25.  19.7 28.2 26.5 14.6 43.1
#  19.6 12.7 29.6 34.9  8.8 33.2 19.7 16.8 20.4 17.8 13.4 15.7 33.4  5.
#  17.7 25.  13.9 20.6 24.   6.3 24.1 28.4 21.2 21.4 20.  16.6 23.1 27.5
#  12.3 20.  50.  16.8 36.2  5.6 17.6 22.4 17.1 33.8  8.5 30.3 19.9 15.1
#  10.4 35.4 20.1 17.5 35.1 19.3  9.5 24.3 19.4 16.3 23.  24.6 24.1 21.2
#  19.4 10.8 27.9 37.9  7.4 20.4 18.7 13.8 32.7 13.9 15.4 17.9 22.6 14.
#  32.5  9.6 16.7 50. ]
# y_predict :
#  [21.4 22.3  9.8 17.8 20.  10.8 13.6 15.2 20.3 24.2 10.7 25.8 33.7 24.5
#  13.4 27.1 20.8 27.8 21.7 18.9 21.9 32.6 28.4 18.5 30.8 24.9 13.1 35.2
#  18.3 14.8 26.6 32.5  9.2 32.5 21.3 18.3 20.8 19.5 13.1 17.8 31.2  8.8
#  18.2 20.5 12.6 21.5 22.6 12.7 22.3 31.5 21.3 20.6 21.6 20.5 14.1 27.3
#  12.9 21.1 40.2 20.7 33.3 10.1 18.8 21.5 20.8 31.9 14.3 29.5 19.5 15.1
#   9.4 32.8 21.4 20.1 31.8 21.3 12.5 21.4 20.7 11.9 19.2 23.3 23.8 22.2
#  20.6 12.  17.7 29.   9.8 21.4 19.6 17.3 31.7 15.5 15.9 10.6 22.5 15.
#  24.6 12.3 17.  29.7]
# RMSE :  4.0446163879486665
# R2 :  0.7903853746852583