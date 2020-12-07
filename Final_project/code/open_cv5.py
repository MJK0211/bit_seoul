import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('./Final_project/images/3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3,5)
imgNum  = 0

for (x,y,w,h) in faces:    
    if x is None:
        print("okokokoo")
    else:
        cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
        cv2.imwrite("./Final_project/photo/" +"aaa" + ".jpg", cropped)
        imgNum += 1 
