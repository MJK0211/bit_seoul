# 이미지에 대한 생성 옵션 정하기
# 어떤 옵션으로 데이터를 바꿀 것인지

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
train_datagen = ImageDataGenerator(rescale=1./255, #정규화
                                   horizontal_flip=True, #수평
                                   vertical_flip=True, #수직
                                   width_shift_range=0.1, #수평이동
                                   height_shift_range=0.1, #수직이동
                                   rotation_range=5,
                                   zoom_range=1.2,
                                   shear_range=0.7, #좌표보정, 좌표이동
                                   fill_mode='nearest' #옮겼을때 빈자리를 그전과 비슷하게 채워준다
)

test_datagen = ImageDataGenerator(rescale=1./255) #테스트는 기존 이미지로 테스트해야하기 때문에 정규화만 한다

# flow 또는 flow_from_directory
# flow: 폴더 아닌곳에서, flow_from_directory : 폴더에서 데이터를 가져옴

xy_train = train_datagen.flow_from_directory(
    './data/img/data1/train', #폴더 위치 
    target_size=(150,150), #이미지 크기 설정 - 기존 이미지보다 크게 설정하면 늘려준다 
    batch_size=5, #5섯장씩 
    class_mode='binary' #클래스모드는 찾아보기!
    # save_to_dir='./data/img/data1_2/train' #전환된 이미지 데이터 파일을 이미지 파일로 저장
) 

# x=(150,150,1), train 폴더안에는 ad/normal이 들어있다. y - ad:0, normal:1

xy_test = test_datagen.flow_from_directory(
    './data/img/data1/test',
    target_size=(150,150),
    batch_size=5,  
    class_mode='binary'
    # save_to_dir='./data/img/data1_2/test'
)

print("=================================================================")
# print(type(xy_train)) #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>

# print(xy_train[0].shape)
# AttributeError: 'tuple' object has no attribute 'shape'
# x, y데이터가 tuple로 저장되있음

# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(xy_train[0][0].shape) #(5, 150, 150, 3) - x
# print(xy_train[0][1].shape) #(5,) - y

# print(xy_train[1][0].shape) #(5, 150, 150, 3) - x   [][]중 앞에 [] 는 batch_size,
# print(xy_train[1][1].shape) #(5,) - y

# print(len(xy_train)) #32 전체 갯수 = len()*batch_size

# np.save('./data/npy/keras63_train_x.npy', arr=xy_train[0][0])
# np.save('./data/npy/keras63_train_y.npy', arr=xy_train[0][1])
# np.save('./data/npy/keras63_test_x.npy', arr=xy_test[0][0])
# np.save('./data/npy/keras63_test_y.npy', arr=xy_test[0][1])
# 데이터 numpy로 변환시 batch_size를 최대로 잡아놓고 저장한다!

# print(xy_train[0][0].shape) #(160, 150, 150, 3)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(20, (2,2), padding='same', input_shape=(150, 150, 3))) 
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['acc'])
from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min') 

hist = model.fit_generator(
    xy_train, #train 안에 x, y의 데이터를 모두 가지고 있다
    steps_per_epoch=32, #
    epochs=50,
    validation_data=xy_test, #test도 x, y의 데이터를 모두 가지고 있다
    validation_steps=24 #어떤건지 찾아보기
)

#4. 평가, 예측
loss, acc = model.evaluate(xy_test)
print("loss : ", loss)
print("acc : ", acc)

# 결과값
# loss :  0.6672096252441406
# acc :  0.6416666507720947