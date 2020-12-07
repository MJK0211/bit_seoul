# -*- coding: utf-8 -*-
import urllib
from bs4 import BeautifulSoup as bs
from urllib.parse import urlencode, quote_plus, unquote
from urllib.request import urlopen, urlretrieve
import urllib
import os

base_url = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query='  
plusUrl = input('검색어 입력: ') 
url = base_url + quote_plus(plusUrl) + '%EC%B6%9C%EC%97%B0%EC%A7%84' 
html = urlopen(url)
soup = bs(html, "html.parser")
name = soup.find("div", class_="list_image_info _content").find_all("li")

find_imglist= list()
find_casting = list()
find_namelist = list()

for item in name:
    find_name = item.find_all(class_="_text")[1] #주인공 이름
    find_namelist.append(find_name.get_text())
    # print(find_namelist)
    find_img = item.find(class_='item').find_all(class_='thumb')
    for j in find_img:
            img = j.find('img')
            find_imglist.append(img.get('src'))
            find_casting.append(img.get('alt'))
            
    for i in range(len(find_imglist)):
        path = './Final_project/images/'+str(i)+'/'
        os.makedirs(path, exist_ok=True)   
        urlretrieve(find_imglist[i], path + find_casting[i]+'.jpg')
        
print(find_namelist)