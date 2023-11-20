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
    """
    比較適合remote sencing
    """
    contours, _ = cv2.findContours(im_in, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 輪廓之個數
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
    消除孤立點
    args:
        threshold_area: 大於保留，小於去除
    return:
        resMatrix: 消除孤立點的圖
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
    #20230608最好版本
    
    # 先將圖片轉成灰階
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #設kernel給dilate and erosion
    kernel = np.ones((3,3),np.uint8) 
    
    #用模糊的方式沒有比較好，會消除掉線條
    # blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
    # blur = cv2.medianBlur(gray_img, 5)
    
    #設一個binary來找出顏色比較不同的地方，通常為陸地亮的部分
    #將找出來的特徵變得更明顯
    (t, binary) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary,kernel,iterations = 20)

    #使用canny來找線條，因為陸地的線條相較於海面會很明顯
    #找出來之線條使用膨脹來加強
    low_threshold = 1
    high_threshold = 100
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)
    dilate = cv2.dilate(edges,kernel,iterations = 20)
    
    # dilate = fill(dilate)
    
    #將dilate和binary的結果相加
    #去除小碎塊面積
    #做erosion來變回原本的形狀
    output = cv2.add(dilate, binary)
    dilate = remove_small_points(dilate, 100000)
    erosion = cv2.erode(output,kernel,iterations = 25)
    
    
    # erosion = fill(erosion)
    
    #如果有船隻被去除，就恢復，設一個一般在遙感影像船隻最大可能的解析度
    thresh1 = erosion > 0
    res_img1 = remove_small_points(thresh1, 70*70)
    res_img1 = fill(res_img1)
    
    
    #設一個新的kernel，5*5可以有更明顯的效果，要填滿沒被填滿的區域
    #因為使用傳統的填滿，會導致整張圖變成一個顏色，因為遙感影像太多碎塊
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(res_img1,kernel,iterations = 20)
    res_img1 = cv2.erode(dilate,kernel,iterations = 20)
    
    #MASK與原圖結合
    # res_img1 = cv2.bitwise_not(res_img1)
    # final = cv2.bitwise_and(img, img, mask=res_img1)

    return res_img1

def stand(image):
    # 讀取影像
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    # gray_img = cv2.medianBlur(gray_img, 5)
    
    window_size = 5

    
    # 計算影像的行數和列數
    rows, cols = gray_img.shape
    row_indices = np.arange(rows - window_size + 1)
    col_indices = np.arange(cols - window_size + 1)
    
    window_shape = (window_size, window_size)
    window_strides = gray_img.strides
    windows = np.lib.stride_tricks.as_strided(
        gray_img,
        shape=(row_indices.size, col_indices.size) + window_shape,
        strides=(window_strides[0], window_strides[1]) + window_strides
    )
    
    # 計算局部統計方差
    variances = np.var(windows, axis=(2, 3))
    result = np.zeros((rows, cols), dtype=np.float32)
    
    # 將方差給對應位置
    result[row_indices[:, np.newaxis], col_indices] = variances
    
    thresholded_image = np.where(result > 10, 255, 0)
    
    # 正規化輸出才有值
    result_normalized = cv2.normalize(thresholded_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    result_normalized = remove_small_points(result_normalized, 70*70)
    
    

    return result_normalized, variances

def otsu_process(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret1, th1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    res_img1 = remove_small_points(th1, 70*70)
    return res_img1

def hist(img):
    max_pixel_values = []  
    max_pixel_frequencies = []  
    pixel_frequencies = []
    
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = ['blue','springgreen','red'] 
    for i in [0,1,2]:
        #計算三個通道pixel values值方圖
        hist = cv2.calcHist([img],[i], None, [256], [0.0,255.0])  
        
        #最高頻率之pixel value
        max_pixel_value = np.argmax(hist)
        max_pixel_frequency = np.max(hist)
        
        #最高頻率次數
        max_pixel_values.append(max_pixel_value)
        max_pixel_frequencies.append(max_pixel_frequency)
        
        #0~51的pixel頻率出現比值
        pixel_frequency = np.sum(hist[0:51]) / np.sum(hist)
        pixel_frequencies.append(pixel_frequency)
        
        plt.subplot(121), plt.plot(hist, color[i])
        plt.title('Histrogram of Color image')

    plt.subplot(122), plt.imshow(RGB_img), plt.title('res_img1')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    
    
    # for i, channel in enumerate(color):
    #     print(f"Max pixel value in {channel} channel: {max_pixel_values[i]}")
    #     print(f"Frequency of max pixel value in {channel} channel: {max_pixel_frequencies[i]}")
    #     print(f"Frequency of pixels in {channel} channel between 0 and 50: {pixel_frequencies[i]}")
    
    # plt.show()
    return max_pixel_values, max_pixel_frequencies, pixel_frequencies

def final_process(img):
    '''
    最終影像mask合併
    '''
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    standard, variances = stand(img) #標準差
    # print('variances: ',variances)
    canny = segFunc(img) #線條
    output1 = cv2.add(standard, canny) #合併
    
    otsuthre = otsu_process(img)
    output2 = cv2.add(otsuthre, canny) #合併
    
    kernel = np.ones((3,3),np.uint8) 
    
    #開, 填滿, 閉
    opening1 = cv2.morphologyEx(output1, cv2.MORPH_OPEN, kernel, iterations=2)
    fill_img1 = fill(opening1)
    closing1 = cv2.morphologyEx(fill_img1, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    opening2 = cv2.morphologyEx(output2, cv2.MORPH_OPEN, kernel, iterations=2)
    fill_img1 = fill(opening2)
    closing2 = cv2.morphologyEx(fill_img1, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    output_final = cv2.add(closing1, closing2) #合併
    #陸地mask通常都會是白色，但需要保留的是海，需要做not
    output = cv2.bitwise_not(output_final)
    
    # plt.subplot(321), plt.imshow(RGB_img, cmap='gray'), plt.title('RGB_img')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(322), plt.imshow(standard, cmap='gray'), plt.title('standard')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(323), plt.imshow(canny, cmap='gray'), plt.title('Canny')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(324), plt.imshow(output, cmap='gray'), plt.title('final')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(325), plt.imshow(otsuthre, cmap='gray'), plt.title('otsu')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(326), plt.imshow(output2, cmap='gray'), plt.title('otsu + canny')
    # plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.show()
    
    return output

def imgProcess(filePath):
    #只要是以下副檔名都可以被讀取
    extensions = tuple(['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo'])
    for filename in os.listdir(filePath):
        
        if filename.endswith(extensions):
            print(filename)
            dir_path = os.path.dirname(filePath) + "/sealand"
            
            img = cv2.imread(filePath + "/" + filename)
            
            #計算值方圖，來判斷是否有包含陸地
            max_pixel_values, max_pixel_frequencies, pixel_frequencies = hist(img)
            
            #blue, green, red
            BMP = max_pixel_values[0]
            GMP = max_pixel_values[1]
            RMP = max_pixel_values[2]
            RMF = max_pixel_frequencies[2]
            RF = pixel_frequencies[2]
            
            #如果pixel value最大值出現在50以下
            #且紅色大於藍綠
            #就判定為陸地
            if RMP <= 50 and GMP <= 50 and BMP <= 50:
                if RMP >= GMP and RMP >= BMP:
                    
                    output = final_process(img)
                    
                    zero_pixels = np.count_nonzero(output == 0)
                    total_pixels = img.shape[0] * img.shape[1]
                    zero_ratio = zero_pixels / total_pixels
                    if cv2.countNonZero(output) == 0:
                        output = cv2.bitwise_not(output)

                        
                    if  zero_ratio >= 0.8:
                        # print('Sea RF: ', RF)
                        if os.path.exists(dir_path) == False:
                            os.makedirs(dir_path)
                        cv2.imwrite(dir_path + "/" + filename, img)
                    
                    else:
                        # print('Land RF: ', RF)
                        final = cv2.bitwise_and(img, img, mask=output)
                        if os.path.exists(dir_path) == False:
                            os.makedirs(dir_path)
                        cv2.imwrite(dir_path + "/" + filename, final)
                        
                else:
                    # print('Sea RF: ', RF)
                    if os.path.exists(dir_path) == False:
                        os.makedirs(dir_path)
                    cv2.imwrite(dir_path + "/" + filename, img)
            
            #如果紅色出現在0~51的頻率占三分之二以上
            #或是介於0.01~0.15
            #就判定為海洋
            elif (RF > 0.66) or (RF <= 0.15 and RF > 0.01):
                # print('Sea RF: ', RF)
                if os.path.exists(dir_path) == False:
                    os.makedirs(dir_path)
                cv2.imwrite(dir_path + "/" + filename, img)
         
            
            else:
                
                output = final_process(img)
                
                zero_pixels = np.count_nonzero(output == 0)
                total_pixels = img.shape[0] * img.shape[1]
                zero_ratio = zero_pixels / total_pixels
                if cv2.countNonZero(output) == 0:
                    output = cv2.bitwise_not(output)

                    
                if  zero_ratio >= 0.8:
                    # print('Sea RF: ', RF)
                    if os.path.exists(dir_path) == False:
                        os.makedirs(dir_path)
                    cv2.imwrite(dir_path + "/" + filename, img)
                
                else:
                    # print('Land RF: ', RF)
                    final = cv2.bitwise_and(img, img, mask=output)
                    if os.path.exists(dir_path) == False:
                        os.makedirs(dir_path)
                    cv2.imwrite(dir_path + "/" + filename, final)


                


