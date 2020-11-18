# OneHotEncodeing

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

 

(x_train, y_train), (x_test, y_test) = mnist.load_data()

 

print(x_train.shape, x_test.shape)  # (60000,28,28), (10000,28,28)

print(y_train.shape, y_test.shape)  # (60000, )      (10000, )

print(x_train[0])

print(y_train[0])

 

# plt.imshow(x_train[0], 'gray')

# plt.show()

 

# 데이터 전처리 1. OneHotEncoding

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)

print(y_train[0])

 

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.

x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

 

print(x_train[0])

 

#2. 모델

 

################### 1. load_model ########################

#3. 컴파일, 훈련

from tensorflow.keras.models import load_model

model1 = load_model('./save/model_test02_2.h5')

#4. 평가, 예측

result1 = model1.evaluate(x_test, y_test, batch_size=32)

print("loss : ", result1[0])

print("accuracy : ", result1[1])

 

############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model

model2 = load_model('./model/mnist-02-0.0512.hdf5')

#4. 평가, 예측

result2 = model2.evaluate(x_test, y_test, batch_size=32)

print("loss : ", result2[0])

print("accuracy : ", result2[1])

 

################ 3. load_weights ##################

# 2. 모델

model3 = Sequential()

model3.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))

model3.add(Conv2D(20, (2,2), padding='valid'))

model3.add(Conv2D(30, (3,3)))

model3.add(Conv2D(40, (2,2), strides=2))

model3.add(MaxPooling2D(pool_size=2))

model3.add(Flatten())

model3.add(Dense(100, activation='relu'))

model3.add(Dense(10, activation='softmax'))

# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

 

model3.load_weights('./save/weight_test02.h5')

 

#4. 평가, 예측

result3 = model3.evaluate(x_test, y_test, batch_size=32)

print("loss : ", result3[0])

print("accuracy : ", result3[1])

 

# loss :  0.06116842105984688

# accuracy :  0.9871000051498413

 

# loss :  0.0450650192797184

# accuracy :  0.9854000210762024

 

# loss :  0.06116842105984688

# accuracy :  0.9871000051498413
[출처] model weight save load|작성자 게마