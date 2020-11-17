#DNN - load_boston

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

from sklearn.preprocessing import MinMaxScaler, StandardScaler #데이터 전처리 StandardScaler 추가

scaler = StandardScaler()
scaler.fit(x)
x_standard = scaler.transform(x) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_standard, y, train_size=0.8) 

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
# loss :  9.415366172790527
# y_test :  [10.9 22.5 35.1 14.9 31.1 23.4 18.9 29.1  7.4 26.5 30.5 24.1 18.3 13.4
#  19.4 23.8 50.  13.1 33.3 18.  17.4 33.1 20.  11.9 20.1 22.9 20.9 20.6
#  23.8 18.5 24.  13.6 19.5 19.1 19.8 24.5 23.8 24.3 21.5 20.1 13.5 21.7
#  24.4 24.8 13.8 12.8 19.6 17.4 24.1 22.8 48.5 22.9 26.7 18.7 23.1 22.2
#  23.1 50.  22.7 25.  18.6 24.8 26.6  5.  22.8 15.6 20.8 20.7 21.7 22.9
#  20.  22.8 16.5 13.1 17.8  8.4 23.  33.4 50.  13.8 13.8 31.7 20.6 25.
#  12.7 16.3 19.  15.7 17.2  7.  32.5 35.2 24.4 20.3 24.4 22.6 21.1 20.6
#  29.6 19.5 20.3 30.3]

# y_predict :
#  [13.149729  21.623878  33.030907  15.798201  32.27707   23.758835
#  22.248951  30.13409   12.992464  22.191402  31.234867  23.803925
#  19.192604  14.345945  21.425932  22.733347  41.763107  15.11353
#  39.956196  18.341936  19.678875  31.14925   19.957111  19.29367
#  22.039549  23.67741   20.393127  21.142185  22.735058  20.22613
#  22.093641  16.864801  18.63184   15.775607  21.45173   22.604914
#  22.201584  22.589823  21.523642  19.498709  14.328426  22.520054
#  22.978241  24.14636   15.721636  13.505299  21.417088  19.450127
#  24.342312  23.54948   41.008186  27.342808  26.589489  19.482231
#  25.173264  23.673456  19.313599  42.79924   20.705215  23.470596
#  20.66753   29.861841  30.446106  11.597537  22.58917   15.409453
#  22.966223  23.169085  21.179672  22.609379  19.643084  22.902721
#  22.201591  12.122576  15.186605  13.468273  19.068995  32.27111
#  41.677097  20.20874   14.6986065 33.665092  22.13418   21.65481
#  14.8622265 14.885566  20.355463  16.81305   15.071589  13.523801
#  29.64832   27.894169  22.939196  19.6899    27.729279  22.654165
#  22.520384  22.343075  31.65964   19.880154  20.120901  35.50786  ]

# RMSE :  1.4995469693232195
# R2 :  0.8585726327076172