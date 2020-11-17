#load-boston - CNN

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.datasets import load_boston #load_boston 주택가격 데이터셋 추가

#1. 데이터
# Attribute Information (in order):
#         - CRIM     범죄율
#         - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#         - INDUS    비소매상업지역 면적 비율
#         - CHAS     찰스강의 경계에 위치 유무 (1:찰스강 경계에 존재, 0:찰스강 경계에 존재X)
#         - NOX      일산화질소 농도
#         - RM       주거 당 평균 방의 갯수
#         - AGE      1940년 이전에 건축된 주택의 비율
#         - DIS      직업 센터의 거리
#         - RAD      방사형 고속도로까지의 거리
#         - TAX      재산세율
#         - PTRATIO  학생/교사 비율
#         - B        인구 중 흑인 비율
#         - LSTAT    인구 중 하위 계층 비율
#         - MEDV     소유주가 거주하는 주택의 가치 ($ 1000 이내)

dataset = load_boston()
x = dataset.data #(506,13)
y = dataset.target #(506,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler #데이터 전처리 StandardScaler 추가

scaler = StandardScaler()
scaler.fit(x)
x_standard = scaler.transform(x) 
x_standard = x_standard.reshape(506,13,1,1)

print(x_standard.shape) #(506,13,1,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_standard, y, train_size=0.8) 

print(x_train.shape) #(404,13,1,1) 

#2. 모델구성 
model = Sequential()
model.add(Conv2D(10, (1,1), padding='same', input_shape=(13,1,1))) 
model.add(Dropout(0.2))
model.add(Conv2D(20, (1,1), padding='valid'))
model.add(Dropout(0.2))
model.add(Conv2D(30, (1,1))) 
model.add(Dropout(0.2))
model.add(Conv2D(40, (1,1), strides=1))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=1)) 
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min') 

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

print("y_test : ", y_test)
print("y_predict : \n", y_predict.reshape(102,))


from sklearn.metrics import mean_absolute_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결과값
# loss :  606.2200317382812
# y_test :  [ 8.8 22.4 18.8 20.1 37.2 24.4 15.2 41.7 16.6 16.8 18.2 19.4 15.4 11.9
#  50.  30.1 20.2 41.3 29.1 10.4 12.1 21.8 27.9 24.7 13.1 20.1 21.4 22.7
#  33.1 28.1 22.5 23.2 17.1 19.5 30.1 22.2 20.  50.  20.9  7.5 14.1 25.1
#  36.2 30.5 20.9 17.8 20.1 22.  16.8 18.5  9.5 16.7 23.8 19.2 24.1 16.4
#  21.4 13.3 21.2 13.  50.  18.4 17.4 27.1 50.  21.2 24.1 38.7 17.5 20.8
#  21.7 19.8 17.5 19.1 20.2 33.1 13.3 43.8 21.  44.8 21.  20.  34.9 20.1
#  25.2 27.  24.  15.  20.4 21.7 16.1 28.7 32.5 22.8 23.5 31.  21.2 48.3
#  28.2 29.8 29.  16.1]
# y_predict :
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1.]
# RMSE :  4.76979877900073
# R2 :  -5.841216800625715