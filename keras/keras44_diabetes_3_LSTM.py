#load-diabetes - LSTM

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM
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
x_standard = x_standard.reshape(442,10,1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_standard, y, train_size=0.8) 

#2. 모델 구성
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(10,1)))
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

from sklearn.metrics import mean_absolute_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결과값
# loss :  4017.01123046875
# y_test :  [184. 246. 109.  64. 341. 110. 209. 162. 281.  84. 321.  90.  93. 281.
#  225. 277. 147. 215. 128. 225. 208. 201. 308. 135. 270.  59.  47. 259.
#  129. 303. 198. 142. 252. 281.  52. 217. 128. 186. 192. 113.  98. 214.
#   31. 170. 145. 111. 202. 123. 139.  65.  72.  64. 102.  55.  39. 197.
#   60. 173. 134.  65. 121.  59. 277.  73. 245. 107. 252. 210. 178.  77.
#   95. 265.  90. 293. 195. 128.  68.  65. 101.  58.  55. 235. 101.  72.
#  179. 102. 118.  79. 200.]
# y_predict :
#  [ 79.03213  164.42857  149.85857   64.33827  245.40466  127.0287
#  110.63489  180.63994  240.327    223.23572  258.45154  131.78435
#   71.185135 242.81873  127.05049  206.52501   67.772064 201.35938
#  260.18106  213.35503  242.67085  113.446594 242.99832  103.84203
#  265.10236   39.975163  73.766815 153.85463   70.78561  237.81424
#  165.20532   90.45555  256.12177  153.10098   65.067604 273.40555
#  111.42293  267.157    176.23352  126.82555  121.60724  103.15917
#  147.41116   85.8149   143.01329  157.54224  193.11324  254.44322
#  125.569435  80.62533  180.63615   82.1034   125.03631   78.8438
#   68.734406 178.57231   67.73897  122.24907  136.62294   60.78441
#  207.89226  119.789345 256.47565  154.09688  179.35097  113.60264
#  175.57323  191.4494   237.53204   43.96124   91.32841  156.67973
#   77.68639  195.94243  253.39337   69.181046 152.32367   41.105206
#  196.44699  151.28654   74.40599  130.25078   81.060844  50.667053
#  185.06755  128.49274   72.06448  107.79131   84.62415 ]
# RMSE :  7.103005816179061
# R2 :  0.37270435172519023