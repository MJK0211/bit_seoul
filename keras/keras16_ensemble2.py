#tran_test_split에서 추가하지 않는다

#1. 데이터
import numpy as np

x1 = np.array([range(1,101), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열 input_shape (3,)
x2 = np.array([range(4,104), range(761,861), range(100)]) 

y1 = np.array([range(101,201), range(311,411), range(100)]) #output 3
y2 = np.array([range(501,601), range(431,531), range(100,200)])
y3 = np.array([range(501,601), range(431,531), range(100,200)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)

x2 = np.transpose(x2)
y2 = np.transpose(y2)

y3 = np.transpose(y3)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.7)
x2_train, x2_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x2, y2, y3, train_size=0.7)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate #concatenate 추가

#성능은 다 같다 but 대문자, 소문자는 사용방법이 다르다
#from tensorflow.keras.layers import concatenate 
#from keras.layers.merge import Concatenate , concatenate - 이전버전
#from keras.layers import Concatenate, concatenate - 이전버전

#2_1 모델1
input1 = Input(shape=(3,)) #input1 layer 구성
dense1_1 = Dense(10, activation='relu', name='king1')(input1) #model.summary() dense부분에 가독성을 위해
dense1_2 = Dense(7, activation='relu', name='king2')(dense1_1) 
dense1_3 = Dense(5, activation='relu', name='king3')(dense1_2)
output1 = Dense(3, name='king4')(dense1_3)

#2_2 모델2
input2 = Input(shape=(3,)) #input2 layer 구성
dense2_1 = Dense(10, activation='relu')(input2) 
dense2_2= Dense(7, activation='relu')(dense2_1) 
dense2_3= Dense(5, activation='relu')(dense2_2)
output2 = Dense(3)(dense2_3)


#모델 병합
# merge1 = concatenate([output1, output2]) #소문자 concatenate
# merge1 = Concatenate(axis=1)([output1, output2]) #대문자 Concatenate, axis는 찾아서 정리한다
merge1 = Concatenate()([output1, output2]) 

middle1 = Dense(3)(merge1)
middle2 = Dense(7)(middle1)
middle3 = Dense(11)(middle2) #layer마다 이름을 같게 해도 상관없다 가독성을 위해 1,2,3표기

#output 모델 구성 (분기-나눔)
output1_1 = Dense(30)(middle3)
output1_2 = Dense(13)(output1_1)
output1_3 = Dense(3)(output1_2)

output2_1 = Dense(30)(middle3)
output2_2 = Dense(13)(output2_1)
output2_3 = Dense(7)(output2_2)
output2_4 = Dense(3)(output2_3)

output3_1 = Dense(30)(middle3)
output3_2 = Dense(13)(output3_1)
output3_3 = Dense(7)(output3_2)
output3_4 = Dense(3)(output3_3)

#모델 정의
model = Model(inputs=[input1, input2], outputs=[output1_3, output2_4, output3_4])

#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=1, validation_split=0.25, verbose=1)

#4. 평가, 예측

result = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=1)

print(result)
y1_pred, y2_pred, y3_pred = model.predict([x1_test, x2_test])

from sklearn.metrics import mean_absolute_error
def RMSE(y1_test, y2_test, y3_test, y1_pred, y2_pred, y3_pred):
    return np.sqrt((mean_absolute_error(y1_test, y1_pred)+mean_absolute_error(y2_test, y2_pred)+mean_absolute_error(y3_test, y3_pred))/3)
print("RMSE : ", RMSE(y1_test, y2_test, y3_test, y1_pred, y2_pred, y3_pred))

from sklearn.metrics import r2_score
r2 = (r2_score(y1_test, y1_pred) + r2_score(y2_test, y2_pred) + r2_score(y3_test, y3_pred))/3
print("R2 : ", r2)

# 결과값
# [6.218577861785889, 1.30415940284729, 2.1856870651245117, 2.7287323474884033, 1.30415940284729, 2.1856870651245117, 2.7287323474884033]
# RMSE :  1.0976331872051985
# R2 :  0.9974463720322414