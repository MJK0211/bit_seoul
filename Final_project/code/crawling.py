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

find_namelist = list()
for item in name:
    find_name = item.find_all(class_="_text")[1] #주인공 이름
    find_namelist.append(find_name.get_text())
print(find_namelist)

for i in range(4):
    base_imgUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='    
    crawl_num = 500
    url_img = base_imgUrl + quote_plus(find_namelist[i])
    print(url_img+" : "+find_namelist[i])    
    html_img = urlopen(url_img)
    soup_img = bs(html_img, "html.parser")
    img = soup_img.find_all(class_='_img')

    path = './Final_project/images/'+str(i)+'/'
    os.makedirs(path, exist_ok=True)
    
    n = 1    
    for j in img:
        # print(n)              
        imgUrl = j['data-source']       
        print(imgUrl)   
        urlretrieve(imgUrl, path + find_namelist[i] + str(n)+'.jpg')
        n += 1
        if n > crawl_num:           
            break               
print('Image Crawling is done.')
