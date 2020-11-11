#tran_test_split에서 추가하지 않는다 #r2 튠

#1. 데이터
import numpy as np

x1 = np.array([range(1,101), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열 input_shape (3,)
x2 = np.array([range(4,104), range(761,861), range(100)]) 
y1 = np.array([range(101,201), range(311,411), range(100)]) #output 3

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, train_size=0.7)
y1_train, y1_test = train_test_split(y1, train_size=0.7)


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
middle3 = Dense(11)(middle2)

#output 모델 구성 (분기-나눔)
output1_1 = Dense(30)(middle3)
output1_2 = Dense(13)(output1_1)
output1_3 = Dense(3)(output1_2)


#모델 정의
model = Model(inputs=[input1, input2], outputs=output1_3)

#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size=1, validation_split=0.25, verbose=1)

#4. 평가, 예측

result = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)

print(result)

y1_pred = model.predict([x1_test, x2_test])

print("y_pred : \n", y1_pred)

from sklearn.metrics import mean_absolute_error
def RMSE(y1_test, y1_pred):
    return np.sqrt(mean_absolute_error(y1_test, y1_pred))
print("RMSE : ", RMSE(y1_test, y1_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y1_pred)
print("R2 : ", r2)

# 결과값
# [1136.2276611328125, 1136.2276611328125]
# y_pred : 
#  [[140.56548  346.89343   46.274776]
#  [143.51445  353.7429    47.24593 ]
#  [146.07222  359.6587    48.0786  ]
#  [146.6075   360.84216   48.231853]
#  [140.07396  345.7519    46.112934]
#  [141.64676  349.40494   46.630875]
#  [149.10522  366.3643    48.946903]
#  [139.01854  343.35452   45.769714]
#  [144.59569  356.2542    47.601974]
#  [140.6638   347.12186   46.307163]
#  [141.94167  350.0898    46.727974]
#  [142.72803  351.91635   46.986927]
#  [144.89058  356.9392    47.699093]
#  [147.3211   362.41986   48.43613 ]
#  [142.23656  350.7748    46.82508 ]
#  [148.03474  363.99762   48.640457]
#  [140.46716  346.6652    46.242428]
#  [141.15527  348.26334   46.46903 ]
#  [148.21318  364.39215   48.69156 ]
#  [144.10417  355.1126    47.440094]
#  [144.98886  357.16748   47.731472]
#  [146.78587  361.23648   48.28292 ]
#  [141.84334  349.86148   46.695595]
#  [147.67793  363.20868   48.5383  ]
#  [138.73604  342.7242    45.67878 ]
#  [142.82634  352.14468   47.01932 ]
#  [143.02292  352.6012    47.084057]
#  [143.90758  354.656     47.375393]
#  [145.08717  357.39578   47.76382 ]
#  [141.45016  348.9483    46.566143]]
# RMSE :  5.515454229557543
# R2 :  0.005399093543403251