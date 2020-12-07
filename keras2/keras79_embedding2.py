import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

docs = ["너무 재밌어요", "참 최고에요", "참 잘 만든 영화에요",
        "추천하고 싶은 영화입니다.", "한번 더 보고 싶네요", "글쎄요",
        "별로에요", "생각보다 지루해요", "연기가 어색해요",
        "재미없어요", "너무 재미없다", "참 재밌네요"] # X값
    
#1 긍정 : 1, 부정 : 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1]) # Y값

token  = Tokenizer()
token.fit_on_texts(docs)

# print(token.word_index)
# {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에 
# 요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23}

x = token.texts_to_sequences(docs)
print(len(x))
# 자연어처리는 시계열이다
# 순서가 있는 문장이다
# 의미있는 숫자를 뒤로 밀고
# 0을 채운다
'''
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre') #앞에 채운다 : pre, 뒤로 채운다 : post

print(pad_x.shape) #(12, 4)
# [[ 0  0  2  3]
#  [ 0  0  1  4]
#  [ 1  5  6  7]
#  [ 0  8  9 10]
#  [11 12 13 14]
#  [ 0  0  0 15]
#  [ 0  0  0 16]
#  [ 0  0 17 18]
#  [ 0  0 19 20]
#  [ 0  0  0 21]
#  [ 0  0  2 22]
#  [ 0  0  1 23]]

word_size = len(token.word_index)+1
print("전체 토큰 사이즈: ", word_size) # 24

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten #임베딩 레이어 

#2. 모델 구성

model = Sequential()
model.add(Embedding(24, 10, input_length=4)) 
# 2가지 방식으로 구성된다. 
# 첫번째로 단어사전의 갯수, 두번째로 output 노드의 갯수 - 임의로 정해도됨, 세번째 (12,4)-> input_length=4
# 임베딩 레이어에서는 첫번째 - 단어의갯수보다 작은값을 넣으면 터진다. 두번째 output 노드의 갯수는 상관없음, 세번째 input_length는 맞춰서 명시하거나 생략가능!

# WARNING:tensorflow:Model was constructed with shape (None, 6) for input Tensor("embedding_input:0", shape=(None, 6), dtype=float32), 
# but it was called on an input with incompatible shape (None, 4). 
# input_length는 컬럼의 갯수이지만 다르게 해도 돌아가긴 한다 but warning 뜸 되도록 맞춰주기

# 기존의 input_length를 명시를 해주고, 안해주고의 파라미터의 갯수 차이는 없다.
# dim을 자동으로 인식을 해준다

# model.add(Embedding(24, 10)) 
# 이거와 Flatten()사용할경우
# ValueError: The last dimension of the inputs to `Dense` should be defined. Found `None`.

# model.add(LSTM(32))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 4, 10)             240
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                5504
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 5,777
# Trainable params: 5,777
# Non-trainable params: 0
# _________________________________________________________________


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min') 
model.fit(pad_x, labels, epochs=30, batch_size=1, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(pad_x, labels, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(pad_x)
y_predict = np.round(y_predict.reshape(12,))
print("y_real : ", labels)
print("y_pred : ", y_predict)

# 결과값
# loss :  0.17932116985321045
# acc :  0.9166666865348816
# y_real :  [1 1 1 1 1 0 0 0 0 0 0 1]
# y_pred :  [1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1.]
'''