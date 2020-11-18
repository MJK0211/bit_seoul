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
x_test = x_test[10:] #(9990,28,28)
y_real = y_test[:10] #(10,)
y_test = y_test[10:] #(9990,)

#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(60000, 10)
print(y_train[0]) #5 - [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] - 0/1/2/3/4/5/6/7/8/9 - 5의 위치에 1이 표시됨

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. #CNN은 4차원이기 때문에 4차원으로 변환, astype -0 형변환
x_test = x_test.reshape(9990,28,28,1).astype('float32')/255.
x_predict = x_predict.reshape(10,28,28,1).astype('float32')/255

#2. 모델
 
################### 1. load_model ########################

#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/mnist_CNN_model_fit.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result1[0])
print("accuracy : ", result1[1]) 

############## 2. load_model ModelCheckPoint #############

model2 = load_model('./model/mnist_CNN-02-0.0562.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result2[0])
print("accuracy : ", result2[1])
 
################ 3. load_weights #########################

# 2. 모델
model3 = Sequential()
model3.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1))) #(28,28,10)
model3.add(Conv2D(20, (2,2), padding='valid')) #(27,27,20)
model3.add(Conv2D(30, (3,3))) #(25,25,30)
model3.add(Conv2D(40, (2,2), strides=2)) #(24,24,40)
model3.add(MaxPooling2D(pool_size=2)) #(12,12,40)
model3.add(Flatten())
model3.add(Dense(100, activation='relu')) 
model3.add(Dense(10, activation='softmax')) 
# model.summary()

# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/mnist_CNN_model_weight.h5') 

#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result3[0])
print("accuracy : ", result3[1])

# 결과값

# 313/313 [==============================] - 0s 2ms/step - loss: 0.0607 - acc: 0.9838
# loss :  0.060665640980005264
# accuracy :  0.9837837815284729

# 313/313 [==============================] - 0s 1ms/step - loss: 0.0487 - acc: 0.9835
# loss :  0.0486760251224041
# accuracy :  0.9834834933280945

# 313/313 [==============================] - 0s 2ms/step - loss: 0.0607 - acc: 0.9838
# loss :  0.060665640980005264
# accuracy :  0.9837837815284729