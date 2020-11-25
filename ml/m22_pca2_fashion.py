# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# fashion_mnist dnn 과 loss / acc를 비교

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import fashion_mnist #dataset인 fashion_mnist
from sklearn.decomposition import PCA

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train_test = np.append(x_train, x_test, axis=0)
print(x_train_test.shape) #(70000, 28, 28)

x_train_test = x_train_test.reshape(x_train_test.shape[0],
                                    x_train_test.shape[1]*x_train_test.shape[2])


pca = PCA()
pca.fit(x_train_test)
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components1 = np.argmax(cumsum >=0.95) + 1 #188
n_components2 = np.argmax(cumsum >=1) + 1 #784
# print(n_components1)
# print(n_components2)

pca = PCA(n_components=188)
x2d = pca.fit_transform((x_train_test))
x_train = x2d[:60000]
x_test = x2d[60000:69990]
x_pred = x2d[69990:]

y_real = y_test[9990:]
y_test = y_test[:9990]

#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성 
model = Sequential()
model.add(Dense(200, input_shape=(188,))) 
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu')) 
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu')) 
model.add(Dense(10, activation='softmax')) 
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
# loss :  0.42182376980781555
# acc :  0.8763763904571533
# y_real :  [5 6 8 9 1 9 1 8 1 5]
# y_pred :  [5 0 8 9 1 9 1 8 1 5]