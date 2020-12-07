#전이학습
#잘만든 모델을 사용하겠다 - 가중치까지!
#메인모델로 훈련은 하지 않지만, 일부만 훈련하겠다
#ImageNet이라는 이미지를 훈련시킨 가중치를 가지고 있다

from tensorflow.keras.applications import VGG16 #VGG16추가
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential

# vgg16 = VGG16() #layer가 16개라서 VGG16
# vgg16 = VGG16(weights='imagenet') #layer가 16개라서 VGG16


# (None, 224, 224, 3) 
# input_shape가 고정되있다.

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3)) #최소 32,32 를 사용해야하고, 채널은 3이상을 써야함
# 따라서 include_top=False를 통해 input_shape를 재정의 한다

# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0

# print(model.trainable_weights)
# array([[[[ 4.29470569e-01,  1.17273867e-01,  3.40129584e-02, ...,
#           -1.32241577e-01, -5.33475243e-02,  7.57738389e-03],
#          [ 5.50379455e-01,  2.08774377e-02,  9.88311544e-02, ...,
#           -8.48205537e-02, -5.11389151e-02,  3.74943428e-02],
#          [ 4.80015397e-01, -1.72696680e-01,  3.75577137e-02, ...,
#           -1.27135560e-01, -5.02991639e-02,  3.48965675e-02]],
# 레이어당 Weight와 Bias가 출력됨

# vgg16.trainable=False
# print(len(vgg16.trainable_weights))
# vgg16.summary()

# vgg16.trainable=False #imagenet을 훈련시키지 않겠다.

# vgg16.summary()
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688

# print(model.trainable_weights)
# []

model = Sequential()
model.add(vgg16)
# model.add(Flatten()) #Dense 모델을 사용할 경우 차원을 맞추기 위해 Flatten()사용
model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

model.summary()
 
print(len(vgg16.trainable_weights)) # 0
print(len(model.trainable_weights)) # 6

# 과적합피하기
#1. 훈련데이더 늘린다
#2. 피처임포턴스
#3. 정규화

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])


print(aaa)


#                                                                            Layer Type   Layer Name  Layer Trainable
# 0  <tensorflow.python.keras.engine.functional.Functional object at 0x000001D823EFB520>  vgg16       True
# 1  <tensorflow.python.keras.layers.core.Flatten object at 0x000001D829F40C40>           flatten     True
# 2  <tensorflow.python.keras.layers.core.Dense object at 0x000001D829F40F10>             dense       True
# 3  <tensorflow.python.keras.layers.core.Activation object at 0x000001D829F73100>        activation  True
# 4  <tensorflow.python.keras.layers.core.Dense object at 0x000001D829F7DE80>             dense_1     True
# 5  <tensorflow.python.keras.layers.core.Dense object at 0x000001D829F86F70>             dense_2     True
