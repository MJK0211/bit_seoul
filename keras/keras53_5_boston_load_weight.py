import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.datasets import load_boston #dataset인 load_boston 추가

#1. 데이터
dataset = load_boston()
x = dataset.data #(506,13), 인덱스 0~12의 값
y = dataset.target #(506,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.8) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler #데이터 전처리 StandardScaler 추가
scaler = StandardScaler()
scaler.fit(x_train)
x_train_standard = scaler.transform(x_train) 
x_test_standard = scaler.transform(x_test)

#2. 모델
 
################### 1. load_model ########################

#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/boston_DNN_model_fit.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test_standard, y_test, batch_size=32)
print("loss : ", result1[0])
print("accuracy : ", result1[1]) 

############## 2. load_model ModelCheckPoint #############
model2 = load_model('./model/boston_DNN-91-6.1769.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test_standard, y_test, batch_size=32)
print("loss : ", result2[0])
print("accuracy : ", result2[1])
 
################ 3. load_weights #########################

#2. 모델 구성
model3 = Sequential()
model3.add(Dense(200, activation='relu', input_shape=(13,)))
model3.add(Dropout(0.2))
model3.add(Dense(180, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(150, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(110, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(60, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(10, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(1))
# model.summary()

# 3. 컴파일
model3.compile(loss='mse', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/boston_DNN_model_weight.h5') 

#4. 평가, 예측
result3 = model3.evaluate(x_test_standard, y_test, batch_size=32)
print("loss : ", result3[0])
print("accuracy : ", result3[1])

# 결과값

# model1 = load_model('./save/boston_DNN_model_fit.h5')
# loss :  10.03355598449707
# accuracy :  0.0

# model2 = load_model('./model/boston_DNN-91-6.1769.hdf5')
# loss :  6.421498775482178
# accuracy :  0.0

# model3.load_weights('./save/boston_DNN_model_weight.h5') 
# loss :  10.03355598449707
# accuracy :  0.0
