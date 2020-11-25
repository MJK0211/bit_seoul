# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# cifar100 dnn 과 loss / acc를 비교

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import cifar100 #dataset인 cifar100
from sklearn.decomposition import PCA

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train_test = np.append(x_train, x_test, axis=0)
print(x_train_test.shape) #(60000, 32, 32, 3)


x_train_test = x_train_test.reshape(x_train_test.shape[0],
                                    x_train_test.shape[1]*x_train_test.shape[2]*x_train_test.shape[3])


pca = PCA()
pca.fit(x_train_test)
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components1 = np.argmax(cumsum >=0.95) + 1 #202
n_components2 = np.argmax(cumsum >=1) + 1 #3072
# print(n_components1)
# print(n_components2)

pca = PCA(n_components=202)
x2d = pca.fit_transform((x_train_test))
x_train = x2d[:50000]
x_test = x2d[50000:59990]
x_pred = x2d[59990:]

y_real = y_test[9990:]
y_test = y_test[:9990]

#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성 
model = Sequential()
model.add(Dense(200, input_shape=(202,))) 
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu')) 
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu')) 
model.add(Dense(100, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min') 
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_pred) # np.argmax로 변환하지 않아도 같은 값으로 출력이 가능하다
y_pred = np.argmax(y_pred, axis=1)
print("y_real : ", y_real.reshape(10,))
print("y_pred : ", y_pred)

# 결과값
# loss :  4.605478763580322
# acc :  0.01001000963151455
# y_real :  [49 33 72 51 71 92 15 14 23  0]
# y_pred :  [8 8 8 8 8 8 8 8 8 8]