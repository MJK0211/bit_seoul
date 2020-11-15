import numpy as np  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

# #2. 모델구성
# model = Sequential()
# model.add(LSTM(200, activation='relu', input_shape=(3,1)))
# model.add(Dense(180, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(110, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(10, activation='relu'))
# # model.add(Dense(1))



input1 = Input(shape=(3,1))
lstm_layer = LSTM(200, activation='relu', name='lstm_layer')(input1)
output1 = Dense(1, name='output1')(lstm_layer) 
model = Model(inputs = input1, outputs = output1) 

# dense1 = Dense(180, activation='relu', name='dense1')(lstm_layer)
# dense2 = Dense(150, activation='relu', name='dense2')(dense1)
# dense3 = Dense(110, activation='relu', name='dense3')(dense2)
# dense4 = Dense(60, activation='relu', name='dense4')(dense3)
# dense5 = Dense(10, activation='relu', name='dense5')(dense4)
# output1 = Dense(1, name='output1')(dense5) 
# model = Model(inputs = input1, outputs = output1) 

# model.summary()
# model.save("./save/keras28_1.h5") 

