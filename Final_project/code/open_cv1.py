import sys, cv2 
import matplotlib.pyplot as plt
print(cv2.__version__) #4.4.0


# img = cv2.imread('./Final_project/photo/cat.jpg', cv2.IMREAD_GRAYSCALE) #filename, flags=None , cv2.IMREAD_GRAYSCALE

imgBGR = cv2.imread('./Final_project/photo/cat.jpg')
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

# if img is None:
#     print('Im load failed!')
#     sys.exit()

# print(type(img)) #<class 'numpy.ndarray'>
# print(img.shape) # (442, 391, 3)
# print(img.dtype) #uint8

# cv2.imwrite(filename, img, params=None) 
# filename:저장할 영상 파일이름(문자열)
# img : 저장할 영상 데이터(numpy.ndarray)
# params : 파일저장 옵션 지정(속성 & 값의 정수의 쌍)
# ex) JPG파일 압축률을 90%로 하고싶다! -> [cv2.IMWRITE_JPEG_QUALITY, 90] 으로 설정
# retval : 정상적으로 저장하면 True, 실패하면 False

# cv2.namedWindow('image')
# cv2.imshow('image', img) 
# cv2.waitKey()
# cv2.destroyAllWindows()
# while True:
#   if cv2.waitKey() == 27: #ESC
#     break

# plt.axis('off')
# plt.imshow(imgRGB)
# plt.show()

# 그레이 스케일 영상 출력
imgGray = cv2.imread('./Final_project/photo/cat.jpg', cv2.IMREAD_GRAYSCALE)
# plt.axis('off')
# plt.imshow(imgGray, cmap='gray')
# plt.show()

# 두개의 영상을 함께 출력

plt.subplot(121), plt.axis('off'), plt.imshow(imgRGB)
plt.subplot(122), plt.axis('off'), plt.imshow(imgGray, cmap='gray')
plt.show()