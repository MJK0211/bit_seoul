import numpy as np
import pandas as pd
# np.set_printoptions(threshold=np.inf, linewidth=np.inf) #inf = infinity 

# #1. 데이터
# ######################################비트 2018/05/04 ~ 2020/11/19
# df1 = pd.read_csv('./data/csv/비트컴퓨터 1120.csv',
#                   index_col=0,
#                   header=0,
#                   encoding='cp949',
#                   sep=',')  #(1200,12)
                  
# bit = df1.sort_values(['일자'], ascending=['True'])
# bit = bit.iloc[573:1199, [0,1,2,3,7]]               

# for i in range(len(bit.index)):
#     for j in range(len(bit.iloc[i])):
#         bit.iloc[i,j] = int(bit.iloc[i,j].replace(',',''))

# #print(bit.shape) #(626, 5)
# bit = bit.to_numpy()
# np.save('./data/npy/bit.npy', arr=bit)

# ######################################삼성 2018/05/04 ~ 2020/11/19
# df2 = pd.read_csv('./data/csv/삼성전자 1120.csv',   
#                   index_col=0,
#                   header=0,
#                   encoding='cp949',
#                   sep=',') #(660,12)
# samsung = df2.sort_values(['일자'], ascending=['True'])
# samsung = samsung.iloc[33:659, [0,1,2,3,7,8]]
# for i in range(len(samsung.index)):
#     for j in range(len(samsung.iloc[i])):
#         samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',',''))

# #print(samsung.shape) #(626, 6)
# samsung = samsung.to_numpy()
# np.save('./data/npy/samsung.npy', arr=samsung)
# ##################################################################

samsung = np.load('./data/npy/samsung.npy', allow_pickle=True).astype('float32')
bit = np.load('./data/npy/bit.npy', allow_pickle=True).astype('float32')

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    x_predict = list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset)+1:
            x_predict = dataset[i:x_end_number, :]          
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 1]
       
        x.append(tmp_x)
        y.append(tmp_y)
        x_predict.append(x_predict)
    return np.array(x), np.array(y), np.array(x_predict)

x_samsung, y_samsung, x_samsung_predict = split_xy5(samsung, 5, 1)
x_bit, y_bit, x_bit_predict = split_xy5(bit, 5, 1)

# print(x_samsung.shape) #(621, 5, 6)
# print(y_samsung.shape) #(621, 1)
# print(x_samsung_predict.shape) #(5, 6)
# print(x_bit.shape) #(621, 5, 5)
# print(y_bit.shape) #(621, 1)
# print(x_bit_predict.shape) #(5, 5)

from sklearn.model_selection import train_test_split
x_samsung_train, x_samsung_test, y_samsung_train, y_samsung_test, = train_test_split(x_samsung, y_samsung, train_size=0.8) 
x_bit_train, x_bit_test, y_bit_train, y_bit_test, = train_test_split(x_bit, y_bit, train_size=0.8) 

x_samsung_train = x_samsung_train.reshape(496,30)
x_samsung_test = x_samsung_test.reshape(125,30)
x_samsung_predict = x_samsung_predict.reshape(1,30)
x_bit_train = x_bit_train.reshape(496,25)
x_bit_test = x_bit_test.reshape(125,25)
x_bit_predict = x_bit_predict.reshape(1,25)

#1.1 삼성데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler1.fit(x_samsung_train)
x_samsung_train_minmax = scaler1.transform(x_samsung_train)
x_samsung_test_minmax = scaler1.transform(x_samsung_test)
x_samsung_predict_minmax = scaler1.transform(x_samsung_predict)

#1.1 비트데이터 전처리
scaler2 = MinMaxScaler()
scaler2.fit(x_bit_train)
x_bit_train_minmax = scaler2.transform(x_bit_train)
x_bit_test_minmax = scaler2.transform(x_bit_test)
x_bit_predict_minmax = scaler2.transform(x_bit_predict)

# print(x_samsung_train_minmax.shape) #(496, 30)
# print(x_samsung_test_minmax.shape) #(125, 30)
# print(x_samsung_predict_minmax.shape) #(1, 30)
# print(x_bit_train_minmax.shape) #(496, 25)
# print(x_bit_test_minmax.shape) #(125, 25)
# print(x_bit_predict_minmax.shape) #(1, 25)

# y_samsung_train = y_samsung_train.reshape(496,)
# y_samsung_test = y_samsung_test.reshape(125,)
# y_bit_train =y_bit_train.reshape(496,)
# y_bit_test =y_bit_test.reshape(125,)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input, LSTM

#2_1 모델1
input1 = Input(shape=(30,)) #input1 layer 구성
dense1_1 = Dense(100, activation='relu')(input1)
dense1_2 = Dense(30, activation='relu')(dense1_1) 
dense1_3 = Dense(7, activation='relu')(dense1_2)
output1 = Dense(1)(dense1_3)

#2_2 모델2
input2 = Input(shape=(25,)) #input2 layer 구성
dense2_1 = Dense(100, activation='relu')(input2) 
dense2_2= Dense(30, activation='relu')(dense2_1) 
dense2_3= Dense(7, activation='relu')(dense2_2)
output2 = Dense(1)(dense2_3)

#모델 병합
merge1 = Concatenate()([output1, output2]) 

middle1 = Dense(30)(merge1)
middle2 = Dense(7)(middle1)
middle3 = Dense(11)(middle2)

#output 모델 구성 (분기-나눔)
output1_1 = Dense(30)(middle3)
output1_2 = Dense(13)(output1_1)
output1_3 = Dense(7)(output1_2)
output1_4 = Dense(1)(output1_3)

#모델 정의
model = Model(inputs=[input1, input2], outputs=output1_4)
model.summary()

#3. 컴파일, 훈련
modelpath = './model/samsung_pred-{epoch:02d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_loss', patience=300, mode='min') 
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min') #val_loss가 가장 좋은 값을 저장할 것이다
model.fit([x_samsung_train_minmax, x_bit_train_minmax], y_samsung_train, epochs=10000, batch_size=32, validation_split=0.25, verbose=1, callbacks=[early_stopping, check_point])

model.save_weights('./save/samsung_pred_model_weight.h5')

#4. 평가, 예측
loss, acc = model.evaluate([x_samsung_test_minmax, x_bit_test_minmax], y_samsung_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict([x_samsung_predict_minmax, x_bit_predict_minmax])
print("11/20 삼성 종가 : ", y_pred)

# 결과값
# loss :  971966.875
# acc :  0.0
# 11/20 삼성 종가 :  [[64190.375]]