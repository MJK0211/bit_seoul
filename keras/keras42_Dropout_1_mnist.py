#Dropout - Mnist_CNN

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout #Dropout 추가
from tensorflow.keras.datasets import mnist 

#1. 데이터
#mnist를 통해 손글씨 1~9까지의 데이터를 사용

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape) #(60000,28,28), (10000,28,28)
# print(y_train.shape, y_test.shape) #(60000,), (10000,)

x_predict = x_test[:10]
y_col = y_test[:10]
# print(x_predict.shape) #(10,28,28)


#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
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
model.add(Dropout(0.2)) #100개의 노드가 있다고 한다면 80개만 사용하겠다
model.add(Conv2D(20, (2,2), padding='valid')) #(27,27,20)
model.add(Dropout(0.2))
model.add(Conv2D(30, (3,3))) #(25,25,30)
model.add(Dropout(0.2))
model.add(Conv2D(40, (2,2), strides=2)) #(24,24,40)
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2)) 
model.add(Flatten()) 
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) 
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 10)        50
# _________________________________________________________________
# dropout (Dropout)            (None, 28, 28, 10)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 27, 27, 20)        820
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 27, 27, 20)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 25, 25, 30)        5430
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 25, 25, 30)        0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 12, 12, 40)        4840
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 12, 12, 40)        0
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 6, 6, 40)          0
# _________________________________________________________________
# dropout_4 (Dropout)          (None, 6, 6, 40)          0
# _________________________________________________________________
# flatten (Flatten)            (None, 1440)              0
# _________________________________________________________________
# dropout_5 (Dropout)          (None, 1440)              0
# _________________________________________________________________
# dense (Dense)                (None, 100)               144100
# _________________________________________________________________
# dropout_6 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1010
# _________________________________________________________________
# dropout_7 (Dropout)          (None, 10)                0
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                110
# =================================================================
# Total params: 156,360
# Trainable params: 156,360
# Non-trainable params: 0
# _________________________________________________________________


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min') 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True) 
model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_predict)
y_pred = np.argmax(y_pred, axis=1) 
print("y_col : ", y_col)
print("y_pred : ", y_pred)

# 결과값
# loss :  0.17690396308898926
# acc :  0.9804999828338623
# y_col :  [7 2 1 0 4 1 4 9 5 9]
# y_pred :  [7 2 1 0 4 1 4 9 5 9]