from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_dog = load_img('./data/dog_cat/dog.jpg', target_size=(224,224))
img_cat = load_img('./data/dog_cat/cat.jpg', target_size=(224,224))
img_rian = load_img('./data/dog_cat/rian.jpg', target_size=(224,224))
# plt.imshow(img_cat)
# plt.show()

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_rian = img_to_array(img_rian)


# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_rian = preprocess_input(arr_rian)

# print(arr_dog.shape) #(331, 512, 3) -> (224, 224, 3)
# print(arr_cat.shape) #(442, 391, 3) -> (224, 224, 3)

arr_input = np.stack([arr_dog, arr_cat, arr_rian])
# print(arr_input.shape) #(224, 224, 3)
 
#2. 모델 구성
model = VGG16()
probs = model.predict(arr_input)
print(probs)
print('probs.shape : ', probs.shape)

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions

results = decode_predictions(probs)
print('-------------------------------')
print('result[0] : ', results[0])
print('-------------------------------')
print('result[1] : ', results[1])
print('-------------------------------')
print('result[2] : ', results[2])
print('-------------------------------')
