import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.datasets import load_iris #dataset인 load_iris 추가

#1. 데이터

dataset = load_iris()
x = dataset.data #(150,4)
y = dataset.target #(150,)
#x의 데이터로 세가지 붓꽃 종 중 하나를 찾는 데이터셋이다

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) 
x_pred = x_test[:10]
x_test = x_test[10:]
y_real = y_test[:10]
y_test = y_test[10:]

# print(x_test.shape) #(20, 4)
# print(y_test.shape) #(20,)
# print(x_train.shape) #(120,4)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train) 
x_test_minmax = scaler.transform(x_test)
x_pred_minmax = scaler.transform(x_pred)

x_train_minmax = x_train_minmax.reshape(120,4,1,1)
x_test_minmax = x_test_minmax.reshape(20,4,1,1)
x_pred_minmax = x_pred.reshape(10,4,1,1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
 
################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model

model1 = load_model('./save/iris_CNN_model_fit.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test_minmax, y_test, batch_size=32)
print("loss : ", result1[0])
print("accuracy : ", result1[1]) 

############## 2. load_model ModelCheckPoint #############
model2 = load_model('./model/iris_CNN-55-0.1482.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test_minmax, y_test, batch_size=32)
print("loss : ", result2[0])
print("accuracy : ", result2[1])
 
################ 3. load_weights #########################

#2. 모델 구성
model3 = Sequential()
model3.add(Conv2D(32, (2, 2), padding='same', activation='relu', input_shape=(4, 1, 1)))
model3.add(Dropout(0.2))
model3.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model3.add(Dropout(0.2))
model3.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
model3.add(Dropout(0.2))
model3.add(Flatten())
model3.add(Dense(128, activation='relu'))
model3.add(Dense(3, activation='softmax')) 
#model3.summary()

# 3. 컴파일
model3.compile(loss='mse', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/iris_CNN_model_weight.h5') 

#4. 평가, 예측
result3 = model3.evaluate(x_test_minmax, y_test, batch_size=32)
print("loss : ", result3[0])
print("accuracy : ", result3[1])

# 결과값

# model1 = load_model('./save/iris_CNN_model_fit.h5')
# loss :  0.003022949444130063
# accuracy :  1.0

# model2 = load_model('./model/iris_CNN-55-0.1482.hdf5')
# loss :  0.036167971789836884
# accuracy :  1.0

# model3.load_weights('./save/iris_CNN_model_weight.h5') 
# loss :  3.418649430386722e-05
# accuracy :  1.0