#ModelCheckPoint - 36_minist_2_CNN.py

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist #dataset인 mnist추가

#1. 데이터
#mnist를 통해 손글씨 1~9까지의 데이터를 사용

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape) #(60000,28,28), (10000,28,28)
# print(y_train.shape, y_test.shape) #(60000,), (10000,)

x_predict = x_test[:10]
y_col = y_test[:10]
print(x_predict.shape) #(10,28,28)


#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
#from sklearn.preprocessing import OneHotEncoder #sklearn
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(60000, 10)
print(y_train[0]) #5 - [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] - 0/1/2/3/4/5/6/7/8/9 - 5의 위치에 1이 표시됨

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. 
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.
x_predict = x_predict.reshape(10,28,28,1).astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1))) #(28,28,10)
model.add(Conv2D(20, (2,2), padding='valid')) #(27,27,20)
model.add(Conv2D(30, (3,3))) #(25,25,30)
model.add(Conv2D(40, (2,2), strides=2)) #(24,24,40)
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten()) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(10, activation='softmax')) 
model.summary()

# model.save('./save/model_test01_1.h5')

#3. 컴파일, 훈련
modelpath = './model/mnist_CNN-{epoch:02d}-{val_loss:.4f}.hdf5' # hdf5의 파일, {epoch:02d} - epoch의 2자리의 정수, {val_loss:.4f} - val_loss의 소수넷째자리까지가 네이밍됨

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
early_stopping = EarlyStopping(monitor='val_loss', patience=10) 
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') #val_loss가 가장 좋은 값을 저장할 것이다

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping, check_point])

# 모델 + 가중치까지 저장
model.save('./save/model_test02_2.h5')

# 가중치만 저장
model.save_weights('./save/weight_test02.h5')

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

y_pred = model.predict(x_predict)
y_pred = np.argmax(y_pred, axis=1) 
print("y_col : ", y_col)
print("y_pred : ", y_pred)

