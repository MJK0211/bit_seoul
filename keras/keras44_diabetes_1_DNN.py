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

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_standard = scaler.transform(x) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_standard, y, train_size=0.8) 

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
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min') 

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

#print(y_predict.shape) #(89, 1)

print("y_test : ", y_test)
print("y_predict : \n", y_predict.reshape(89,))

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결과값
# loss :  4908.92626953125
# y_test :  [258. 131. 292.  47. 297. 104. 104. 200. 104. 212.  66. 118.  78. 283.
#  275.  72. 111. 182.  74.  81. 308.  97. 135. 124. 152. 259. 341. 139.
#  137.  92. 310. 107. 150. 179.  90. 225.  60. 113. 200. 167. 249.  78.
#  273.  37. 233. 181. 206. 111. 137. 258. 202. 281.  55. 263. 153. 257.
#  178. 160. 270.  58. 292. 229.  54. 265. 101. 258. 115. 102. 217. 123.
#  268. 107. 103. 129.  61. 110. 222. 127. 233. 281. 146. 288. 121. 275.
#  202. 230.  99.  53.  60.]
# y_predict :
#  [251.05566  161.29413   77.35864   55.151123 249.9555    76.88692
#   97.36992  209.36716  142.95473  192.20685  189.7864   122.370926
#  102.62993  138.29611  242.25018   69.47027   69.55275  152.04189
#  128.89838  223.09802  242.5506   147.38083   51.515324 125.15905
#  135.60971  105.3929   249.36493  145.36243  168.06769   85.3889
#   57.175556  88.236664 159.22745  121.5773   136.77097  121.48695
#   67.38202  181.99765  170.63815  149.3021   172.66138   86.00252
#  181.37674   89.12468  250.13168   59.02523  186.71759  166.96518
#  105.55473  101.094894 198.68845  213.94235   60.108456 234.69678
#   78.142365  82.60757  175.88754  144.04678  221.80496   69.37135
#  231.85452  202.54068   83.06164  113.95727  167.46283  186.86237
#  139.20865   72.64992  213.78976  229.66864  239.93405  206.12683
#  196.0975   136.83707  102.176056 139.09329  222.06647  157.28674
#  214.53989  234.16577  153.65654  243.81     125.64203  238.51727
#  188.84872  179.76187  115.95705   50.45632   65.799225]
# RMSE :  6.977591536575327
# R2 :  0.2486338309611087