from selenium import webdriver

chrome_path = 'd:/bit_down/chromedriver.exe'
browser = webdriver.Chrome(chrome_path)
browser.get('http://www.google.com')