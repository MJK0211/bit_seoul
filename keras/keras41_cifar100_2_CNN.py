#Cifar100 CNN

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.datasets import cifar100 #dataset인 cifar100추가
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_predict = x_test[:10]
x_test = x_test[10:]
y_real = y_test[:10]
y_test = y_test[10:]

print(x_train.shape) #(50000, 32, 32, 3)
print(x_test.shape) #(9990, 32, 32, 3)
print(y_test.shape) #(9990, 100)
print(y_real.shape) #(10,1)

#1_1. 데이터 전처리 - OneHotEncoding

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.astype('float32')/255. 
x_test = x_test.astype('float32')/255.
x_predict = x_predict.astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(Conv2D(20, (2,2), padding='same', input_shape=(32,32,3))) #(32,32,20)
model.add(Dropout(0.2))
model.add(Conv2D(40, (2,2), padding='same')) #(32,32,40)
model.add(Dropout(0.2))
model.add(Conv2D(60, (3,3), padding='same')) #(32,32,60)
model.add(Dropout(0.2))
model.add(Conv2D(80, (2,2), strides=2)) #(16,16,80)
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2)) #(8,8,80)
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min') 
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)
print("y_real : ", y_real.reshape(10,))
print("y_pred : ", y_predict)

# 결과값
# loss :  4.2568206787109375
# acc :  0.35285285115242004
# y_real :  [49 33 72 51 71 92 15 14 23  0]
# y_pred :  [79 80  4 97 71 79 65 63 71  0]