#load-boston - DNN

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from sklearn.datasets import load_boston #load_boston 주택가격 데이터셋 추가

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

# dataset = load_boston()
# x = dataset.data #(506,13), 인덱스 0~12의 값
# y = dataset.target #(506,), 인덱스 13의 값 - 소유주가 거주하는 주택의 가치 (단위 : $ 1000)
# #x의 데이터로 본인의 주택의 가치를 평가하는 데이터 셋이다.

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.8) 

x_train = np.load('./data/npy/boston_x_train.npy')
x_test = np.load('./data/npy/boston_x_test.npy')
y_train = np.load('./data/npy/boston_y_train.npy')
y_test = np.load('./data/npy/boston_y_test.npy')

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

model.save('./save/boston_DNN_model.h5')

#3. 컴파일, 훈련
modelpath = './model/boston_DNN-{epoch:02d}-{val_loss:.4f}.hdf5' # hdf5의 파일, {epoch:02d} - epoch의 2자리의 정수, {val_loss:.4f} - val_loss의 소수넷째자리까지가 네이밍됨
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='min') 
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min')

hist = model.fit(x_train_standard, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping, check_point])

model.save('./save/boston_DNN_model_fit.h5')
model.save_weights('./save/boston_DNN_model_weight.h5')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4. 평가, 예측
result = model.evaluate(x_test_standard, y_test)
print("loss : ", result[0])
print("acc : ", result[1])

y_predict = model.predict(x_test_standard)
print("y_test : ", y_test)
print("y_predict : \n", np.round(y_predict.reshape(102,),1))

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
 
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) #단위 무엇인지 찾아보기
plt.subplot(2,1,1) #2장(2행1열) 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2) #2장(2행1열) 중 두번째
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

# 결과값
# loss :  18.560558319091797
# acc :  0.0
# y_test :  [23.  21.4 27.5 18.6 12.1 22.4 22.6 15.2 13.4 29.9 31.2 23.8 25.  25.
#  13.1 28.6 19.6 24.8 32.9 25.1 11.5 19.9 32.2 10.2 13.4 31.6 20.4 37.9
#  22.2 18.4 15.  22.9 29.  16.7 20.3 11.8  8.5 19.1 24.7 10.4 21.9 46.7
#  19.9 13.1  5.  23.2 15.4 37.2 19.7 50.  22.2 17.2 17.3 14.9 15.7 13.4
#   9.5 34.9 48.3 23.1 23.9  8.7 15.6 30.1 20.5 21.8 20.  14.5 31.5  8.3
#  19.8 44.8 12.5 15.1 22.  43.1 23.2  6.3 16.6 20.5 19.3 23.1 21.1 45.4
#  50.  15.  22.  18.7 11.3 42.3 22.6 17.1 16.2 16.1 20.3 23.9 23.9 24.8
#  27.1 50.  20.5  5. ]
# y_predict :
#  [20.5 25.  10.9 17.6 12.  20.6 21.2 15.   8.9 28.5 26.1 22.6 24.5 22.3
#   9.  27.4 17.3 27.  30.3 23.8 13.4 16.8 29.8  9.7 10.3 28.7 17.3 30.9
#  20.3 15.7 21.2 19.8 30.2 16.7 19.1  8.6 13.8 20.1 20.6 13.8 43.2 40.1
#  18.9 13.5  9.6 23.6 12.1 32.  15.4 43.4 20.5 13.2 15.5 14.  14.6 12.
#   8.9 32.6 40.  21.1 23.   8.4 15.  25.  17.5 20.4 16.9 12.3 31.2 10.1
#  19.4 40.4 15.2 15.4 19.5 34.3 19.8 12.5 14.4 20.7 19.  20.8 19.5 40.5
#  37.6 20.8 18.7 16.7 12.  40.9 25.4 12.1 17.1 18.  18.9 23.9 22.4 22.4
#  25.9 55.5 18.6  8.1]
# RMSE :  4.308196471218565
# R2 :  0.8192074071215594