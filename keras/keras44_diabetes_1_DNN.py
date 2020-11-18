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

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min') 

model.fit(x_train_standard, y_train, epochs=500, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test_standard, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test_standard)

#print(y_predict.shape) #(89, 1)

print("y_test : ", y_test)
print("y_predict : \n", np.round(y_predict.reshape(89,),1))

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결과값
# loss :  4575.6328125
# y_test :  [ 42. 198. 221.  80. 261.  48. 153.  94.  87. 283. 136. 115. 310. 175.
#   37. 258. 184.  94. 210.  42. 113. 102. 258.  93.  78. 259. 145. 229.
#  150. 152.  70.  96. 182. 222.  59.  65.  92. 249.  51. 164. 310. 303.
#  280.  69.  52. 217. 259.  48. 191. 150. 179. 235. 197. 275.  51. 170.
#  109.  53. 346. 277.  72. 259. 257. 275.  49. 104. 297.  40.  85. 262.
#  219. 189.  47. 131. 217.  90.  84. 128. 127.  57. 321. 107. 135.  61.
#  191. 214.  90.  59. 292.]
# y_predict :
#  [109.9 137.6 160.2  83.8 186.6 146.5  82.1 140.8  77.2 163.9 117.   95.1
#  164.7 150.7  62.6 181.1 133.3  77.  111.7  75.7 119.6 105.5 236.6  65.4
#  144.  190.1 119.  152.8 182.5  98.6  55.3  61.2 102.6 168.8 105.2  43.6
#   60.1 152.1 105.4 129.4 236.4 214.3 184.2 103.2  62.6 199.4 197.7  61.6
#   81.  122.8  76.7 141.9 170.8 198.2  92.1  66.  112.6  58.9 214.4 169.5
#   68.5 136.1 164.5 164.4  81.2  63.8 186.7 117.9  84.3 120.7  98.3 156.5
#   90.6 141.6 153.3 112.8 157.5  66.1  88.4  61.7 215.5 101.1  89.6 100.3
#  147.  105.2  95.3  66.7 146.8]
# RMSE :  67.6434253494797
# R2 :  0.3918271111252529