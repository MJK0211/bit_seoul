#전이학습
#잘만든 모델을 사용하겠다 - 가중치까지!
#메인모델로 훈련은 하지 않지만, 일부만 훈련하겠다
#ImageNet이라는 이미지를 훈련시킨 가중치를 가지고 있다

from tensorflow.keras.applications import VGG16 #VGG16추가
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101, ResNet152, ResNet152V2, ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile

models = [VGG16(), VGG19(), Xception(), ResNet101(), ResNet101(), ResNet152(), ResNet152V2(), ResNet50(), ResNet50V2(), 
          InceptionResNetV2(), InceptionV3(), MobileNet(), MobileNetV2(), DenseNet121(), DenseNet169(), DenseNet201(),
          NASNetLarge(), NASNetMobile()]

for i in range(len(models)):
    print("model : ", models[i]._name, "/ params : ", models[i].count_params(), "/ weight : ", len(models[i].trainable_weights))

# model :  vgg16 / params :  138357544 / weight :  32
# model :  vgg19 / params :  143667240 / weight :  38
# model :  xception / params :  22910480 / weight :  156
# model :  resnet101 / params :  44707176 / weight :  418
# model :  resnet101 / params :  44707176 / weight :  418
# model :  resnet152 / params :  60419944 / weight :  622
# model :  resnet152v2 / params :  60380648 / weight :  514
# model :  resnet50 / params :  25636712 / weight :  214
# model :  resnet50v2 / params :  25613800 / weight :  174
# model :  inception_resnet_v2 / params :  55873736 / weight :  490
# model :  inception_v3 / params :  23851784 / weight :  190
# model :  mobilenet_1.00_224 / params :  4253864 / weight :  83
# model :  mobilenetv2_1.00_224 / params :  3538984 / weight :  158
# model :  densenet121 / params :  8062504 / weight :  364
# model :  densenet169 / params :  14307880 / weight :  508
# model :  densenet201 / params :  20242984 / weight :  604
# model :  NASNet / params :  88949818 / weight :  1018
# model :  NASNet / params :  5326716 / weight :  742