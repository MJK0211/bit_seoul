import numpy as np

#1. 데이터

samsung = np.load('./data/npy/samsung.npy', allow_pickle=True).astype('float32')
bit = np.load('./data/npy/bit.npy', allow_pickle=True).astype('float32')
kosdaq = np.load('./data/npy/kosdaq.npy', allow_pickle=True).astype('float32')
gold = np.load('./data/npy/gold.npy', allow_pickle=True).astype('float32')

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    x_predict = list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):            
            tmp_x_predict = dataset[i-1:x_end_number-1, :]        
            x_predict.append(tmp_x_predict)  
            break
        if i>0:
            tmp_x = dataset[i-1:x_end_number-1, :]
            tmp_y = dataset[x_end_number:y_end_number, 0]     
            
            x.append(tmp_x)
            y.append(tmp_y)
            
    return np.array(x), np.array(y), np.array(x_predict)
x_samsung, y_samsung, x_samsung_predict = split_xy5(samsung, 3, 1)
x_bit, y_bit, x_bit_predict = split_xy5(bit, 3, 1)
x_kosdaq, y_kosdaq, x_kosdaq_predict = split_xy5(kosdaq, 3, 1)
x_gold, y_gold, x_gold_predict = split_xy5(gold, 3, 1)

from sklearn.model_selection import train_test_split
x_samsung_train, x_samsung_test, y_samsung_train, y_samsung_test, = train_test_split(x_samsung, y_samsung, train_size=0.8) 
x_bit_train, x_bit_test, y_bit_train, y_bit_test, = train_test_split(x_bit, y_bit, train_size=0.8) 
x_kosdaq_train, x_kosdaq_test, y_kosdaq_train, y_kosdaq_test, = train_test_split(x_kosdaq, y_kosdaq, train_size=0.8) 
x_gold_train, x_gold_test, y_gold_train, y_gold_test, = train_test_split(x_gold, y_gold, train_size=0.8) 

print(x_samsung_train.shape) #(497, 3, 6)
print(x_samsung_test.shape) #(125, 3, 6)
print(x_samsung_predict.shape) #(1, 3, 6)

print(x_samsung.shape)
print(x_bit_train.shape) #(497, 3, 5)
print(x_bit_test.shape) #(125, 3, 5)
print(x_bit_predict.shape) #(1, 3, 5)

print(x_kosdaq_train.shape) #(497, 3, 4)
print(x_kosdaq_test.shape) #(125, 3, 4)
print(x_kosdaq_predict.shape) #(1, 3, 4)

print(x_gold_train.shape) #(497, 3, 3)
print(x_gold_test.shape) #(125, 3, 3)
print(x_gold_predict.shape) #(1, 3, 3)

x_samsung_train = x_samsung_train.reshape(497,18)
x_samsung_test = x_samsung_test.reshape(125,18)
x_samsung_predict = x_samsung_predict.reshape(1,18)

x_bit_train = x_bit_train.reshape(497,15)
x_bit_test = x_bit_test.reshape(125,15)
x_bit_predict = x_bit_predict.reshape(1,15)

x_kosdaq_train = x_kosdaq_train.reshape(497,12)
x_kosdaq_test = x_kosdaq_test.reshape(125,12)
x_kosdaq_predict = x_kosdaq_predict.reshape(1,12)

x_gold_train = x_gold_train.reshape(497,9)
x_gold_test = x_gold_test.reshape(125,9)
x_gold_predict = x_gold_predict.reshape(1,9)

#1.1 삼성데이터 전처리
from sklearn.preprocessing import MinMaxScaler

scaler1 = MinMaxScaler()
scaler1.fit(x_samsung_train)
x_samsung_train_minmax = scaler1.transform(x_samsung_train)
x_samsung_test_minmax = scaler1.transform(x_samsung_test)
x_samsung_predict_minmax = scaler1.transform(x_samsung_predict)

#1.2 비트데이터 전처리
scaler2 = MinMaxScaler()
scaler2.fit(x_bit_train)
x_bit_train_minmax = scaler2.transform(x_bit_train)
x_bit_test_minmax = scaler2.transform(x_bit_test)
x_bit_predict_minmax = scaler2.transform(x_bit_predict)

#1.3 코스닥데이터 전처리
scaler3 = MinMaxScaler()
scaler3.fit(x_kosdaq_train)
x_kosdaq_train_minmax = scaler3.transform(x_kosdaq_train)
x_kosdaq_test_minmax = scaler3.transform(x_kosdaq_test)
x_kosdaq_predict_minmax = scaler3.transform(x_kosdaq_predict)

#1.4 골드데이터 전처리
scaler4 = MinMaxScaler()
scaler4.fit(x_gold_train)
x_gold_train_minmax = scaler4.transform(x_gold_train)
x_gold_test_minmax = scaler4.transform(x_gold_test)
x_gold_predict_minmax = scaler4.transform(x_gold_predict)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input, LSTM

#2_1 모델1
input1 = Input(shape=(18,)) #input1 layer 구성
dense1_1 = Dense(100, activation='relu')(input1)
dense1_2 = Dense(30, activation='relu')(dense1_1) 
dense1_3 = Dense(7, activation='relu')(dense1_2)
output1 = Dense(1)(dense1_3)

#2_2 모델2
input2 = Input(shape=(15,)) #input2 layer 구성
dense2_1 = Dense(100, activation='relu')(input2) 
dense2_2= Dense(30, activation='relu')(dense2_1) 
dense2_3= Dense(7, activation='relu')(dense2_2)
output2 = Dense(1)(dense2_3)

#2_3 모델3
input3 = Input(shape=(12,)) #input3 layer 구성
dense3_1 = Dense(100, activation='relu')(input3) 
dense3_2= Dense(30, activation='relu')(dense3_1) 
dense3_3= Dense(7, activation='relu')(dense3_2)
output3 = Dense(1)(dense3_3)

#2_4 모델4
input4 = Input(shape=(9,)) #input4 layer 구성
dense4_1 = Dense(100, activation='relu')(input4) 
dense4_2= Dense(30, activation='relu')(dense4_1) 
dense4_3= Dense(7, activation='relu')(dense4_2)
output4 = Dense(1)(dense4_3)

#모델 병합
merge1 = Concatenate()([output1, output2, output3, output4]) 

middle1 = Dense(30)(merge1)
middle2 = Dense(7)(middle1)
middle3 = Dense(11)(middle2)

#output 모델 구성 (분기-나눔)
output1_1 = Dense(30)(middle3)
output1_2 = Dense(13)(output1_1)
output1_3 = Dense(7)(output1_2)
output1_4 = Dense(1)(output1_3)

#모델 정의
model = Model(inputs=[input1, input2, input3, input4], outputs=output1_4)
model.summary()


#3. 컴파일, 훈련
#modelpath = './model/samsung_pred-{epoch:02d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2000, mode='min') 
#check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min') #val_loss가 가장 좋은 값을 저장할 것이다
model.fit([x_samsung_train_minmax, x_bit_train_minmax, x_kosdaq_train_minmax, x_gold_train_minmax], y_samsung_train, epochs=50000, batch_size=32, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#model.save_weights('./save/samsung_pred_model_weight.h5')

#4. 평가, 예측
loss, mae = model.evaluate([x_samsung_test_minmax, x_bit_test_minmax, x_kosdaq_test_minmax, x_gold_test_minmax], y_samsung_test, batch_size=32)
print("loss : ", loss)
print("mae : ", mae)

y_pred = model.predict([x_samsung_predict_minmax, x_bit_predict_minmax, x_kosdaq_predict_minmax, x_gold_predict_minmax])
print("11/23 삼성 시가 : ", y_pred)
