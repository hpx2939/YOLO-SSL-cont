# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:53:02 2023

@author: user
"""

import os
import cv2
import time
import numpy as np

from skimage import morphology
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


def holefill_fun(img):
    hole=img.copy()
    cv2.floodFill(hole,None,(0,0),255) # 找到洞孔
    hole=cv2.bitwise_not(hole)

    # or
    filledEdgesOut=cv2.bitwise_or(img,hole)
    # or
    filledEdgesOut=cv2.add(img,hole)

    return filledEdgesOut



def fill(im_in):
    contours, _ = cv2.findContours(im_in, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 80000:
            cv_contours.append(contour)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
        else:
            continue
            
    im_out = cv2.fillPoly(im_in, cv_contours, (255, 255, 255))
    return im_out

def remove_small_points(binary_img, threshold_area):
    """
    消除二值图像中面积小于某个阈值的连通域(消除孤立点)
    args:
        binary_img: 二值图
        threshold_area: 面积条件大小的阈值,大于阈值保留,小于阈值过滤
    return:
        resMatrix: 消除孤立点后的二值图
    """
    #输出二值图像中所有的连通域
    img_label, num = label(binary_img, connectivity=1, background=0, return_num=True) #connectivity=1--4  connectivity=2--8
    # print('+++', num, img_label)
    #输出连通域的属性，包括面积等
    props = regionprops(img_label) 
    ## adaptive threshold
    # props_area_list = sorted([props[i].area for i in range(len(props))]) 
    # threshold_area = props_area_list[-2]
    resMatrix = np.zeros(img_label.shape).astype(np.uint8)
    for i in range(0, len(props)):
        # print('--',props[i].area)
        if props[i].area > threshold_area:
            tmp = (img_label == i + 1).astype(np.uint8)
            #组合所有符合条件的连通域
            resMatrix += tmp 
    resMatrix *= 255
    
    return resMatrix

def segFunc(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    (t, binary) = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    low_threshold = 1
    high_threshold = 75
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)
    
    kernel = np.ones((3,3),np.uint8) 
    dilate = cv2.dilate(edges,kernel,iterations = 20)
    dilate = remove_small_points(dilate, 100000)
    # dilate = fill(dilate)
    output = cv2.add(dilate, binary)
    erosion = cv2.erode(output,kernel,iterations = 20)
    
    
    # erosion = fill(erosion)
    
    thresh1 = erosion > 0
    res_img1 = remove_small_points(thresh1, 2500)
    res_img1 = fill(res_img1)
    
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(res_img1,kernel,iterations = 20)
    res_img1 = cv2.erode(dilate,kernel,iterations = 20)
    # thresh1 = output > 0
    # res_img1 = morphology.remove_small_objects(thresh1, 5000)
    
    res_img1 = cv2.bitwise_not(res_img1)
    
    if cv2.countNonZero(res_img1) == 0:
        res_img1 = cv2.bitwise_not(res_img1)
        
    final = cv2.bitwise_and(img, img, mask=res_img1)
    
    # final = cv2.bitwise_and(img, img, mask=res_img2)
    
    # plt.subplot(321), plt.imshow(RGB_img), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(322), plt.imshow(binary, cmap='gray'), plt.title('binary')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(323), plt.imshow(edges, cmap='gray'), plt.title('Canny')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(324), plt.imshow(output, cmap='gray'), plt.title('add')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(325), plt.imshow(erosion, cmap='gray'), plt.title('erosion')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(326), plt.imshow(res_img1, cmap='gray'), plt.title('res_img1')
    # plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.show()

    return final

def segFunc2(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
    (t, binary) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    low_threshold = 1
    high_threshold = 75
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)
    
    kernel = np.ones((3,3),np.uint8) 
    dilate = cv2.dilate(edges,kernel,iterations = 20)
    dilate = remove_small_points(dilate, 100000)
    # dilate = fill(dilate)
    output = cv2.add(dilate, binary)
    # erosion = cv2.erode(dilate,kernel,iterations = 5)
    
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  
    # opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)  
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) 
    
    thresh1 = output > 0
    res_img1 = morphology.remove_small_objects(thresh1, 2500)
    # res_img1 = res_img1.copy
    
    ## remove_small_objects method2
    res_img2 = remove_small_points(output, 2500)
    res_img2 = cv2.bitwise_not(res_img2)
    # if cv2.countNonZero(res_img2) == 0:
    #     print("Black")
    
    final = cv2.bitwise_and(img, img, mask=res_img2)
    
    # plt.subplot(321), plt.imshow(RGB_img), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(322), plt.imshow(binary, cmap='gray'), plt.title('binary')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(323), plt.imshow(edges, cmap='gray'), plt.title('Canny')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(324), plt.imshow(output, cmap='gray'), plt.title('add')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(325), plt.imshow(res_img1, cmap='gray'), plt.title('res_img1')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(326), plt.imshow(res_img2, cmap='gray'), plt.title('res_img2')
    # plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.show()

    return final

def imgProcess(filePath):
    extensions = tuple(['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo'])
    for filename in os.listdir(filePath):
        
        if filename.endswith(extensions):
            print(filename)
            img = cv2.imread(filePath + "/" + filename)
            final = segFunc(img)
            # segFunc(img)
            path = os.path.dirname(filePath) + "/sealand"
            if os.path.exists(path) == False:
                os.makedirs(path)
            cv2.imwrite(path + "/" + filename, final)



