from urllib.request import urlopen, Request
from urllib.parse import urlencode,unquote,quote_plus
import urllib
from bs4 import BeautifulSoup
import numpy as np

year_data = np.array(['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'])
month_data = np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])

find_list = list()
# 활용승인 절차 개발단계 : 허용 / 운영단계 : 허용
# 신청가능 트래픽 1000000 / 운영계정은 활용사례 등록시 신청하면 트래픽 증가 가능
# 요청주소 http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getAnniversaryInfo
                                                                         #/getRestDeInfo
                                                                         #/getHoliDeInfo
                                                                         #/get24DivisionsInfo

for i in range(len(year_data)):
    for j in range(len(month_data)):
        url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo'
        
        queryParams = '?' + urlencode({ quote_plus('ServiceKey') : 'c7WMSvfEAOgcB4zodbfU6DmHk0QFGL2P5Gto%2FKzEWUFClpktpAm%2Fqr84%2BhX831olWazkCAC4WXzVerCkUSe5Pw%3D%3D', 
        quote_plus('solYear') : year_data[i], quote_plus('solMonth') :  month_data[j]})

        request = urllib.request.Request(url+unquote(queryParams))
        # print ('Your Request:\n'+url+queryParams)
        request.get_method = lambda: 'GET'
        response_body = urlopen(request).read()

        soup = BeautifulSoup(response_body, 'html.parser')
        data = soup.find_all('item')
       
        for item in data:
            locdate = item.find('locdate')  
            find_list.append(locdate.get_text())
    

find_list = np.array(find_list)

print(find_list)
print(type(find_list)) #<class 'numpy.ndarray'>

# 결과값
# ['20110101' '20110202' '20110203' '20110204' '20110301' '20110505'
#  '20110510' '20110606' '20110815' '20110911' '20110912' '20110913'
#  '20111003' '20111225' '20120101' '20120122' '20120123' '20120124'
#  '20120301' '20120411' '20120505' '20120528' '20120606' '20120815'
#  '20120929' '20120930' '20121001' '20121003' '20121219' '20121225'
#  '20130101' '20130209' '20130210' '20130211' '20130301' '20130505'
#  '20131003' '20131225' '20140101' '20140130' '20140131' '20140201'
#  '20140301' '20140505' '20140506' '20140604' '20140606' '20140815'
#  '20140907' '20140908' '20140909' '20140910' '20141003' '20141009'
#  '20141225' '20150101' '20150218' '20150219' '20150220' '20150301'
#  '20150505' '20150525' '20150606' '20150815' '20150926' '20150927'
#  '20160207' '20160208' '20160209' '20160210' '20160301' '20160413'
#  '20160505' '20160514' '20160606' '20160815' '20160914' '20160915'
#  '20160916' '20161003' '20161009' '20161225' '20170101' '20170127'
#  '20170128' '20170129' '20170130' '20170301' '20170503' '20170505'
#  '20170509' '20170606' '20170815' '20171002' '20171003' '20171003'
#  '20171004' '20171005' '20171006' '20171009' '20171225' '20180101'
#  '20180215' '20180216' '20180217' '20180301' '20180505' '20180507'
#  '20180522' '20180606' '20180613' '20180815' '20180923' '20180924'
#  '20180925' '20180926' '20181003' '20181009' '20181225' '20190101'
#  '20190204' '20190205' '20190206' '20190301' '20190505' '20190506'
#  '20190512' '20190606' '20190815' '20190912' '20190913' '20190914'
#  '20191003' '20191009' '20191225' '20200101' '20200124' '20200125'
#  '20200126' '20200127' '20200301' '20200415' '20200430' '20200505'
#  '20200606' '20200815' '20200817' '20200930' '20201001' '20201002'
#  '20201003' '20201009' '20201225']

