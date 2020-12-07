import cv2

#재생할 파일 
VIDEO_FILE_PATH = './Final_project/mp4/startup_13.mp4'

# 동영상 파일 열기
cap = cv2.VideoCapture(VIDEO_FILE_PATH)

#잘 열렸는지 확인
if cap.isOpened() == False:
    print ('Can\'t open the video (%d)' % (VIDEO_FILE_PATH))
    exit()

titles = ['orig']
#윈도우 생성 및 사이즈 변경
for t in titles:
    cv2.namedWindow(t)

#재생할 파일의 넓이 얻기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#재생할 파일의 높이 얻기
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#재생할 파일의 프레임 레이트 얻기
fps = cap.get(cv2.CAP_PROP_FPS)

print('width {0}, height {1}, fps {2}'.format(width, height, fps))

#XVID가 제일 낫다고 함.
#linux 계열 DIVX, XVID, MJPG, X264, WMV1, WMV2.
#windows 계열 DIVX
#저장할 비디오 코덱
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#저장할 파일 이름
filename = './Final_project/mp4/startup_detect.avi'

#파일 stream 생성
out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
#filename : 파일 이름
#fourcc : 코덱
#fps : 초당 프레임 수
#width : 넓이
#height : 높이

#얼굴 인식용
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # 얼굴찾기 haar 파일  

while(True):
    #파일로 부터 이미지 얻기
    ret, frame = cap.read()
    #더 이상 이미지가 없으면 종료
    #재생 다 됨
    if frame is None:
        break;

    #얼굴인식 영상 처리
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur =  cv2.GaussianBlur(grayframe,(5,5), 0)
    faces = face_cascade.detectMultiScale(blur, 1.8, 2, 0, (50, 50))

    #원본 이미지에 얼굴 인식된 부분 표시
    for (x,y,w,h) in faces:
        cx = int(x+(w/2))
        cy = int(y+(h/2))
        cr = int(w/2)
        cv2.circle(frame,(cx,cy),cr,(0,255,0),3)

    # 얼굴 인식된 이미지 화면 표시
    cv2.imshow(titles[0],frame)

    # 인식된 이미지 파일로 저장
    out.write(frame)

    #1ms 동안 키입력 대기
    if cv2.waitKey(1) == 27:
        break;


#재생 파일 종료
cap.release()
#저장 파일 종료
out.release()
#윈도우 종료
cv2.destroyAllWindows()


