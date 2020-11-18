#load-boston - LSTM

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM
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
x_standard = x_standard.reshape(506,13,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_standard, y, train_size=0.8) 

#2. 모델 구성
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(13,1)))
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
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min') 

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
print("y_test : ", y_test)
print("y_predict : \n", y_predict.reshape(102,))

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결과값
# loss :  15.318950653076172
# y_test :  [23.3 25.3 18.  20.9 20.4 12.8 15.  30.5 31.1 23.  50.  48.3 14.4 26.4
#  13.3 22.9  8.7 44.8 15.1 21.  25.  37.3 17.  14.5 19.7 24.2 33.4 16.1
#  19.4 14.1 13.8 19.1 18.6 24.6 20.5 35.4 22.  21.7 27.1 10.5 17.1 32.9
#  17.6 21.4 10.5 16.7 19.  19.5 17.4  8.4 21.2 23.6 20.3 21.2 17.4 22.2
#  23.7 13.4 35.1 15.6 26.6 16.3 13.3 32.  25.  16.6 23.6 20.4 14.5 18.3
#  18.4 19.  17.8 13.9 24.5  7.2 17.4 21.4 15.6 10.9 23.1 21.8 25.  20.5
#  14.6 43.8 13.4 19.9  7.5 22.  23.9 33.8 14.8 50.  50.  27.5 23.1 28.
#  12.7 25.2 30.3 13.6]
# y_predict :
#  [27.049683  24.105814  18.346798  21.097649  18.727657  12.857847
#  19.621746  31.34857   29.500038  28.81105   44.52619   36.325897
#  19.75064   22.006489  15.450889  24.220188  12.847647  37.05398
#  14.754321  18.733294  28.370102  31.670647  17.20013   14.900455
#  21.262701  23.769703  38.331306  23.837273  24.668003  14.7383175
#  15.510811  15.136867  22.236124  24.122217  22.903109  37.77649
#  19.089252  20.515194  19.04488   11.423859  14.731318  34.686264
#  20.478626  43.501534  11.5434675 14.689179  13.952483  20.124046
#  19.559158  14.461807  19.407255  24.158663  22.00159   20.246723
#  19.892551  22.904808  25.733438  13.740685  31.414532  15.684805
#  25.918861  15.4136095 15.054551  31.477314  24.202578  20.256748
#  30.182825  23.23877   14.427981  19.02818   15.525703  18.935669
#  15.897228  14.4095    25.383038  12.768935  16.754368  21.280302
#  15.691803  13.517958  29.153336  21.776222  20.118849  23.09039
#  13.787896  45.500977  12.598475  18.889584  14.158688  23.431896
#  22.63165   33.395813  15.251306  47.149345  47.41947   24.110088
#  19.291708  29.507618  13.895303  24.271395  30.197838  14.307987 ]
# RMSE :  1.6041207738518493
# R2 :  0.8120889427476332