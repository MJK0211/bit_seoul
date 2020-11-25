# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# load_iris dnn 과 loss / acc를 비교

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.datasets import load_iris #dataset인 load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1. 데이터
iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target , train_size=0.8, random_state=42, shuffle=True)

print(x_train.shape) #(120, 4)
print(x_test.shape) #(30, 4)

x_train_test = np.append(x_train, x_test, axis=0)
print(x_train_test.shape) #(150, 4)

# pca = PCA()
# pca.fit(x_train_test)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# n_components1 = np.argmax(cumsum >=0.95) + 1 #2
# n_components2 = np.argmax(cumsum >=1) #3
# print(cumsum)
# print(n_components1)
# print(n_components2) 

pca = PCA(n_components=2)
x2d = pca.fit_transform((x_train_test))
x_train = x2d[:120]
x_test = x2d[120:140]
x_pred = x2d[140:]

y_real = y_test[20:]
y_test = y_test[:20]

#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성 
model = Sequential()
model.add(Dense(200, input_shape=(2,))) 
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu')) 
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu')) 
model.add(Dense(3, activation='softmax')) 
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
print("y_real : ", y_real)
print("y_pred : ", y_pred)

# 결과값
# loss :  0.12302875518798828
# acc :  0.949999988079071
# y_real :  [0 2 0 2 2 2 2 2 0 0]
# y_pred :  [0 2 0 2 2 2 2 2 0 0]