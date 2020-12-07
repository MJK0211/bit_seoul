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

model=Sequential()
model.add(Embedding(10000, 100, input_length=1000))  
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
es=EarlyStopping(monitor='val_loss', patience=10, mode='auto')

model.fit(x_train, y_train, epochs=30, callbacks=[es], validation_split=0.2)
acc=model.evaluate(x_test, y_test)[1]

print('acc : ', acc)