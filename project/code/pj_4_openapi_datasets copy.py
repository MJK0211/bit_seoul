from urllib.request import urlopen, Request
from urllib.parse import urlencode,unquote,quote_plus
import urllib
from bs4 import BeautifulSoup
import numpy as np
import datetime as dt

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

# print(find_list)
# print(type(find_list)) #<class 'numpy.ndarray'>
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


change_day = list()

for i in range(len(find_list)):
    change_day.append(dt.datetime.strptime(find_list[i], '%Y%m%d').strftime('%Y-%m-%d'))

change_days = np.array(change_day)

np.save('./project/data/npy/holiday.npy', arr=change_days)
# print(change_days)
# print(type(change_days))
# ['2011-01-01' '2011-02-02' '2011-02-03' '2011-02-04' '2011-03-01'
#  '2011-05-05' '2011-05-10' '2011-06-06' '2011-08-15' '2011-09-11'
#  '2011-09-12' '2011-09-13' '2011-10-03' '2011-12-25' '2012-01-01'
#  '2012-01-22' '2012-01-23' '2012-01-24' '2012-03-01' '2012-04-11'
#  '2012-05-05' '2012-05-28' '2012-06-06' '2012-08-15' '2012-09-29'
#  '2012-09-30' '2012-10-01' '2012-10-03' '2012-12-19' '2012-12-25'
#  '2013-01-01' '2013-02-09' '2013-02-10' '2013-02-11' '2013-03-01'
#  '2013-05-05' '2013-05-17' '2013-06-06' '2013-08-15' '2013-09-18'
#  '2013-09-19' '2013-09-20' '2013-10-03' '2013-12-25' '2014-01-01'
#  '2014-01-30' '2014-01-31' '2014-02-01' '2014-03-01' '2014-05-05'
#  '2014-05-06' '2014-06-04' '2014-06-06' '2014-08-15' '2014-09-07'
#  '2014-09-08' '2014-09-09' '2014-09-10' '2014-10-03' '2014-10-09'
#  '2014-12-25' '2015-01-01' '2015-02-18' '2015-02-19' '2015-02-20'
#  '2015-03-01' '2015-05-05' '2015-05-25' '2015-06-06' '2015-08-15'
#  '2015-09-26' '2015-09-27' '2015-09-28' '2015-09-29' '2015-10-03'
#  '2015-10-09' '2015-12-25' '2016-01-01' '2016-02-07' '2016-02-08'
#  '2016-02-09' '2016-02-10' '2016-03-01' '2016-04-13' '2016-05-05'
#  '2016-05-14' '2016-06-06' '2016-08-15' '2016-09-14' '2016-09-15'
#  '2016-09-16' '2016-10-03' '2016-10-09' '2016-12-25' '2017-01-01'
#  '2017-01-27' '2017-01-28' '2017-01-29' '2017-01-30' '2017-03-01'
#  '2017-05-03' '2017-05-05' '2017-05-09' '2017-06-06' '2017-08-15'
#  '2017-10-02' '2017-10-03' '2017-10-03' '2017-10-04' '2017-10-05'
#  '2017-10-06' '2017-10-09' '2017-12-25' '2018-01-01' '2018-02-15'
#  '2018-02-16' '2018-02-17' '2018-03-01' '2018-05-05' '2018-05-07'
#  '2018-05-22' '2018-06-06' '2018-06-13' '2018-08-15' '2018-09-23'
#  '2018-09-24' '2018-09-25' '2018-09-26' '2018-10-03' '2018-10-09'
#  '2018-12-25' '2019-01-01' '2019-02-04' '2019-02-05' '2019-02-06'
#  '2019-03-01' '2019-05-05' '2019-05-06' '2019-05-12' '2019-06-06'
#  '2019-08-15' '2019-09-12' '2019-09-13' '2019-09-14' '2019-10-03'
#  '2019-10-09' '2019-12-25' '2020-01-01' '2020-01-24' '2020-01-25'
#  '2020-01-26' '2020-01-27' '2020-03-01' '2020-04-15' '2020-04-30'
#  '2020-05-05' '2020-06-06' '2020-08-15' '2020-08-17' '2020-09-30'
#  '2020-10-01' '2020-10-02' '2020-10-03' '2020-10-09' '2020-12-25']
# <class 'numpy.ndarray'>


