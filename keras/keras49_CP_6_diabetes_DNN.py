#load-diabetes - DNN

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.datasets import load_diabetes #load_diabetes 당뇨병데이터

#1. 데이터
# Attribute Information (in order):
#     0    - age     나이
#     1    - sex     성별
#     2    - bmi     bmi 체질량 지수
#     3    - bp      bp 평균 혈압
#     4    - s1 tc   T- 세포 (백혈구의 일종)
#     5    - s2 ldl  저밀도 지단백질
#     6    - s3 hdl  고밀도 지단백질
#     7    - s4 tch  갑상선 자극 호르몬
#     8    - s5 ltg  라모트리진
#     9    - s6 glu   혈당 수치
#    10    - target  1년 후 질병 진행의 측정

dataset = load_diabetes()
x = dataset.data #(442,10), 인덱스 0~9의 값
y = dataset.target #(442,), 인덱스 10의 값- 1년 후 당뇨병 진행의 측정
#x의 데이터로 1년 후 당뇨병 진행을 측정하는 데이터셋이다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_standard = scaler.transform(x_train) 
x_test_standard = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(10,)))
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

model.save('./save/diabetes_DNN_model.h5')

#3. 컴파일, 훈련
modelpath = './model/diabetes_DNN-{epoch:02d}-{val_loss:.4f}.hdf5' # hdf5의 파일, {epoch:02d} - epoch의 2자리의 정수, {val_loss:.4f} - val_loss의 소수넷째자리까지가 네이밍됨
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min') 
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min')

hist = model.fit(x_train_standard, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping, check_point])

model.save('./save/diabetes_DNN_model_fit.h5')
model.save_weights('./save/diabetes_DNN_model_weight.h5')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4. 평가, 예측
result = model.evaluate(x_test_standard, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

y_predict = model.predict(x_test_standard)
print("y_test : ", y_test)
print("y_predict : \n", np.round(y_predict.reshape(89,),1))

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
# loss :  3101.2724609375
# acc :  0.0
# y_test :  [ 69. 214. 179. 209.  59. 107. 235. 182.  96. 135. 200. 100. 253.  37.
#   51. 174. 144.  68. 138.  86.  72.  89. 124. 147.  78. 118. 111.  90.
#  303. 143.  65. 341. 262. 138.  91. 103. 132.  88.  83. 142. 311. 202.
#  197. 321.  59. 244. 144. 229.  65. 233. 132. 115.  96. 196. 232.  48.
#   92. 197. 137. 261.  58.  90.  72. 186.  77.  85. 173. 252.  53. 110.
#   85. 187. 217.  83. 135. 236.  60. 273. 180.  93. 200. 163. 257. 104.
#   91. 151. 125.  55. 258.]
# y_predict :
#  [120.  129.6 156.9 147.3 154.5 160.9 155.2 125.9  85.3 115.7 128.9 149.5
#  127.6  69.9  68.6 164.1 150.  149.  168.2 154.8  64.  114.1 110.9 161.4
#  182.1  98.7 115.  125.4 239.2  75.9  66.9 236.4 145.3  71.  105.3 159.4
#  125.4 109.3  62.5 133.8 167.4 121.1 152.  231.9 112.7 165.9 142.6 167.2
#   86.1 193.5 228.7 101.2  74.7 176.3 213.9  85.2 118.8 201.7 170.8 212.
#  141.9 161.8  84.  194.9  78.7 137.5 171.9 140.7 124.9 163.8  88.  126.
#  184.4 126.9  96.5 216.  120.6 243.  150.4  82.4 192.1 211.9 170.4  65.4
#  163.8 161.   92.9 161.9 259.7]
# RMSE :  55.68906880321059
# R2 :  0.4278195456121182