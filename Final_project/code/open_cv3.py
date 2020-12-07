#명함 검출

import sys, cv2 
import matplotlib.pyplot as plt

src = cv2.imread('./Final_project/photo/cat.jpg')

src = cv2.resize(src, (0,0), fx=0.5, fy=0.5)
src_RGB = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

plt.subplot(121), plt.axis('off'), plt.imshow(src_RGB)
plt.subplot(122), plt.axis('off'), plt.imshow(src_gray, cmap='gray')
plt.show()

