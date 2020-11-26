from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지에 대한 생성 옵션 정하기
# 어떤 옵션으로 데이터를 바꿀 것인지

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
) 

# x=(150,150,1), train 폴더안에는 ad/normal이 들어있다. y - ad:0, normal:1

xy_test = test_datagen.flow_from_directory(
    './data/img/data1/test',
    target_size=(160,160),
    batch_size=5,  
    class_mode='binary'
)

#3. 훈련
model.fit_generator(
    xy_train, #train 안에 x, y의 데이터를 모두 가지고 있다
    steps_per_epoch=100, #
    epochs=20,
    validation_data=xy_train, #test도 x, y의 데이터를 모두 가지고 있다
    validation_steps=4 #어떤건지 찾아보기
)