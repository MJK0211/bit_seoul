import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2. 모델구성
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(3,1)))
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

model.summary()

model.save("./save/keras28_1.h5") 

