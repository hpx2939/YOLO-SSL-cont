# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:09:47 2023

@author: user
"""
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
import matplotlib.pyplot as plt


# plt.ion()


img = cv2.imread('test.png',1)

# plt.imshow(img)
# plt.title('Title')  # 设置图像标题
# plt.show()
# time.sleep(2)
# plt.pause(0.001)
# plt.close()

# plt.ioff()
cv2.imshow("test1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()