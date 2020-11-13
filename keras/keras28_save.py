import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2. 모델구성
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(4,1)))
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

model.summary()

# model.save("keras28.h5") 
# 모델 저장하기 : 경로는 bit_seoul 폴더 안에 저장되있다. (Default는 Root 폴더) , 파이참은 현재 작업하고있는 폴더가 ROOT가 된다.

model.save("./save/keras28.h5") #저장 폴더 경로 설정, 4개다 저장이 되지만 주의할 것은 '\n'은 개행처리가 되기때문에 오류가 된다. 
# model.save(".\save\keras28_2.h5")
# model.save(".//save//keras28_3.h5")
# model.save(".\\save\\keras28_4.h5")

