import numpy as np
import cv2


# 얼굴과 눈을 검출하기 위해 미리 학습시켜 놓은 XML 포맷으로 저장된 분류기를 로드합니다. 
# face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "./Final_project/haarcascade_frontalface_default.xml")  
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "./Final_project/haarcascade_eye.xml")

# 얼굴찾기 haar 파일  


# 얼굴과 눈을 검출할 그레이스케일 이미지를 준비해놓습니다. 
img = cv2.imread('./Final_project/photo/배수지.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이미지에서 얼굴을 검출합니다. 
faces = face_cascade.detectMultiScale(gray, 1.3, 5)


# 얼굴이 검출되었다면 얼굴 위치에 대한 좌표 정보를 리턴받습니다. 
for (x,y,w,h) in faces:

    # 원본 이미지에 얼굴의 위치를 표시합니다. 
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # 눈 검출은 얼굴이 검출된 영역 내부에서만 진행하기 위해 ROI를 생성합니다. 
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # 눈을 검출합니다. 
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # 눈이 검출되었다면 눈 위치에 대한 좌표 정보를 리턴받습니다. 
    for (ex,ey,ew,eh) in eyes:
        # 원본 이미지에 얼굴의 위치를 표시합니다. ROI에 표시하면 원본 이미지에도 표시됩니다. 
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


# 얼굴과 눈 검출 결과를 화면에 보여줍니다.
cv2.imshow('img',img)
cv2.waitKey(0)

cv2.destroyAllWindows()