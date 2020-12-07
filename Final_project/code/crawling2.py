# -*- coding: utf-8 -*-
import urllib
from bs4 import BeautifulSoup as bs
from urllib.parse import urlencode, quote_plus, unquote
from urllib.request import urlopen, urlretrieve
import urllib
import os
import cv2

base_url = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query='  
plusUrl = input('검색어 입력: ') 
url = base_url + quote_plus(plusUrl) + '%EC%B6%9C%EC%97%B0%EC%A7%84' 

html = urlopen(url)
soup = bs(html, "html.parser")
name = soup.find("div", class_="list_image_info _content").find_all("li")

find_namelist = list()
for item in name:
    find_name = item.find_all(class_="_text")[1] #주인공 이름
    find_namelist.append(find_name.get_text())
print(find_namelist)

crawl_num = 5
for i in range(2):
    base_imgUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='        
    url_img = base_imgUrl + quote_plus(find_namelist[i])
    print(url_img+" : "+find_namelist[i])    
    html_img = urlopen(url_img)
    soup_img = bs(html_img, "html.parser")
    img = soup_img.find_all(class_='_img')

    # path = './Final_project/images/'+str(i)+'/'
    path = './Final_project/images/'

    os.makedirs(path, exist_ok=True)
    
    n = 1    
    for j in img:
        # print(n)              
        imgUrl = j['data-source']          
        # print(imgUrl)
        urlretrieve(imgUrl, path + str(n) + '.jpg')
     
        n += 1
        if n > crawl_num:           
            break               
   
print('Image Crawling is done.')


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


for k in range(crawl_num):
    img = cv2.imread(path + str(k+1) + '.jpg')    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    imgNum  = 0
    for (x,y,w,h) in faces:
        cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
        cv2.imwrite("./Final_project/photo/" + str(imgNum) + ".jpg", cropped)
        imgNum += 1

# img = cv2.imread(path+find_namelist[i] + str(n-1)+'.jpg')


# src = cv2.imread(path+'1.jpg')
# cv2.imshow('Image view', src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2

# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3,5)
# imgNum  = 0
# for (x,y,w,h) in faces:
#     cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
#     # 이미지를 저장
#     cv2.imwrite("./Final_project/photo/sooji" + str(imgNum) + ".jpg", cropped)
#     imgNum += 1
# cv2.imshow('Image view', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
