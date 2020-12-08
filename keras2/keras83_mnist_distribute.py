
################################ GPU 분산처리 ###################################

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()       

print(x_train.shape, x_test.shape)                           
print(y_train.shape, y_test.shape)                           

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical              
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)
print(y_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255.     
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255.

# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce)       # GPU 분산처리 할때 사용

with strategy.scope():
    model = Sequential()
    model.add(Conv2D(20, (2,2), input_shape = (28, 28, 1), padding = 'same'))           
    model.add(Conv2D(40, (2,2), padding = 'valid'))                                
    model.add(Conv2D(10, (3,3)))                                                       
    model.add(Conv2D(20, (2,2), strides = 1))                                      
    model.add(MaxPooling2D(pool_size = 2))                                          
    model.add(Flatten())                                                             
    model.add(Dense(100, activation = 'relu'))                                          
    model.add(Dense(10, activation = 'softmax'))                                       

    model.summary()

# 3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])  
model.fit(x_train, y_train, epochs = 10, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

# 4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 100)
print("loss : ", loss)
print("accuracy : ", accuracy)


