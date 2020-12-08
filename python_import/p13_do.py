import p11_car
import p12_tv

# 운전하다
# car.py의 module 이름은 :  p11_car
# 시청하다
# tv.py의 module 이름은 :  p12_tv

# 자기자신을 불러왔을 때는 __name__ 이 __main__이지만
# 지금처럼 import 했을 시에는 파일 이름이 __name__이 된다

print("===================")
print("do.py의 module 이름은 : ", __name__)
print("===================")

# ===================
# do.py의 module 이름은 :  __main__
# ===================

p11_car.drive()
p12_tv.watch()