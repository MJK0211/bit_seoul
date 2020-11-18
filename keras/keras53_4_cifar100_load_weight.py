import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.datasets import cifar100 #dataset인 cifar100 추가

#1. 데이터
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
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.astype('float32')/255. 
x_test = x_test.astype('float32')/255.
x_predict = x_predict.astype('float32')/255

#2. 모델
 
################### 1. load_model ########################

#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/cifar100_CNN_model_fit.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result1[0])
print("accuracy : ", result1[1]) 

############## 2. load_model ModelCheckPoint #############
from tensorflow.keras.models import load_model
model2 = load_model('./model/cifar100_CNN-14-3.9791.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result2[0])
print("accuracy : ", result2[1])
 
################ 3. load_weights #########################

#2. 모델구성 
model3 = Sequential()
model3.add(Conv2D(20, (2,2), padding='same', input_shape=(32,32,3))) #(32,32,20)
model3.add(Dropout(0.2))
model3.add(Conv2D(40, (2,2), padding='same')) #(32,32,40)
model3.add(Dropout(0.2))
model3.add(Conv2D(60, (3,3), padding='same')) #(32,32,60)
model3.add(Dropout(0.2))
model3.add(Conv2D(80, (2,2), strides=2)) #(16,16,80)
model3.add(Dropout(0.2))
model3.add(MaxPooling2D(pool_size=2)) #(8,8,80)
model3.add(Dropout(0.2))
model3.add(Flatten()) 
model3.add(Dropout(0.2))
model3.add(Dense(200, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(100, activation='softmax')) 
#model.summary()

# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/cifar100_CNN_model_weight.h5') 

#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result3[0])
print("accuracy : ", result3[1])

# 결과값

# model1 = load_model('./save/cifar100_CNN_model_fit.h5')
# loss :  4.3634233474731445
# accuracy :  0.3375375270843506

# model2 = load_model('./model/cifar100_CNN-14-3.9791.hdf5')
# loss :  3.9014499187469482
# accuracy :  0.327727735042572

# model3.load_weights('./save/cifar100_CNN_model_weight.h5') 
# loss :  4.3634233474731445
# accuracy :  0.3375375270843506