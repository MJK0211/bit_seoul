import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.datasets import load_breast_cancer #dataset인 load_breast_cancer 추가

#1. 데이터

dataset = load_breast_cancer()
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

#2. 모델
 
################### 1. load_model ########################

#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/cancer_DNN_model_fit.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test_standard, y_test, batch_size=32)
print("loss : ", result1[0])
print("accuracy : ", result1[1]) 

############## 2. load_model ModelCheckPoint #############
model2 = load_model('./model/cancer_DNN-18-0.0727.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test_standard, y_test, batch_size=32)
print("loss : ", result2[0])
print("accuracy : ", result2[1])
 
################ 3. load_weights #########################

#2. 모델 구성
model3 = Sequential()
model3.add(Dense(32, input_shape=(30,))) 
model3.add(Dropout(0.2))
model3.add(Dense(64, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(128, activation='relu'))
model3.add(Dropout(0.2)) 
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(128, activation='relu')) 
model3.add(Dropout(0.2))
model3.add(Dense(64, activation='relu')) 
model3.add(Dropout(0.2))
model3.add(Dense(1, activation='sigmoid')) 
#model.summary()

# 3. 컴파일
model3.compile(loss='mse', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/cancer_DNN_model_weight.h5') 

#4. 평가, 예측
result3 = model3.evaluate(x_test_standard, y_test, batch_size=32)
print("loss : ", result3[0])
print("accuracy : ", result3[1])

# 결과값

# model1 = load_model('./save/cancer_DNN_model_fit.h5')
# loss :  0.023952463641762733
# accuracy :  0.9912280440330505

# model2 = load_model('./model/cancer_DNN-18-0.0727.hdf5')
# loss :  0.0801176130771637
# accuracy :  0.9561403393745422

# model3.load_weights('./save/cancer_DNN_model_weight.h5') 
# loss :  0.008322995156049728
# accuracy :  0.9912280440330505