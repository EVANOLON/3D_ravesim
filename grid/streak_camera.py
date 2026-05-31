import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

gauss_kernel=[(0.00012341,0.00360656,0.011109,0.00360656,0.00012341),(0.00360656,0.10539922,0.32465247,0.10539922,0.00360656),(0.011109,0.32465247,1,0.32465247,0.011109),(0.00360656,0.10539922,0.32465247,0.10539922,0.00360656),(0.00012341,0.00360656,0.011109,0.00360656,0.00012341)]
kernel=[(-1,-1,-1),(0,3,0),(1,1,1)]

image=cv2.imread('02.jpg')
img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#img=img[200:800,200:800]
lower = np.percentile(img, 5)
upper = np.percentile(img, 95)
#img = np.clip(img, lower, upper)

#中值滤波，可消除孤立噪点
img=scipy.signal.medfilt(img,7)
#高斯卷积核，可降低高斯噪声
#img=scipy.signal.convolve(img,gauss_kernel,mode='valid')
#边缘增强
img=scipy.signal.convolve(img,kernel,mode='valid')

print(img.shape)
plt.imshow(img,'gray')
plt.colorbar()
plt.show()