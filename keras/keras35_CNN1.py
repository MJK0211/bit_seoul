#CNN

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D #2차원 이미지 Conv2D 추가

#1. 데이터
x = np.array([])


#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1))) 

# channels - 흑백은 1, 컬러면 3
# filters - 다음 레이어로 던져주는 갯수(10개) & 정수, 출력 공간의 차원 (예 : 컨볼 루션의 출력 필터 수).
# kernel_size - 2D 컨볼 루션 창의 높이와 너비를 지정하는 정수 또는 2 개 정수의 튜플 / 목록입니다.
# strides=(1, 1) - 기본 이미지를 자를때 1칸씩 자르겠다 

# padding="valid"  "valid"or 중 하나 "same"(대소 문자 구분 안함). "valid"패딩이 없음을 의미합니다. 
#                  "same"출력이 입력과 동일한 높이 / 너비 치수를 갖도록 입력의 왼쪽 / 오른쪽 또는 위 / 아래에 균일하게 패딩됩니다.

# 입력모양(input_shape) = (rows, cols, channels) - 입력모양, 세로(rows), 가로(cols), channels

# shape = batch_size, rows, cols, channels - 100개짜리 batch_size를 10으로 지정하게 된다면 10번씩 10번돈다(10장씩)

# 참고 LSTM
# units = filters와 같다
# return_sequence
# 입력모양 : batch_size, timesteps, feature - timesteps(10일치씩 자르겠다, 시간의 간격의 규칙)
# input_shape = (timesteps, feature)


#3. 컴파일, 훈련






#4. 평가, 예측