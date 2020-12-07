from tensorflow.keras.datasets import reuters, imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(y_train.shape, y_test.shape) #(25000,) (25000,)

# y의 카테고리 갯수 출력
category = np.max(y_train)+1
print('카테고리 : ', category)
# 카테고리 :  2

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)
# [0 1]

x_train=pad_sequences(x_train, maxlen=1000, padding='pre')
x_test=pad_sequences(x_test, maxlen=1000, padding='pre')

print(x_train.shape, x_test.shape) # (25000, 1000) (25000, 1000)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, MaxPooling1D, Bidirectional #임베딩 레이어

model = Sequential()
# model.add(Embedding(2376, 10)) 
model.add(Embedding(2000, 128)) 
# model.add(LSTM(32, input_shape=(4,1)))
# model.add(Flatten())
# model.add(Conv1D(32, 2, input_shape=(4,1)))
# model.add(Flatten()) 
model.add(Conv1D(10, 5, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))

model.add(Bidirectional(LSTM(10)))
# model.add(LSTM(10))
model.add(Dense(1,activation='sigmoid'))
model.summary()

'''
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min') 
model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)
# # y_predict = np.round(y_predict.reshape(46,))
# print("y_real : ", y_test)
# print("y_pred : ", y_predict)
'''