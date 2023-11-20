# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:53:02 2023

@author: user
"""

import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
# from skimage.measure import label, regionprops


def fill(im_in):
    """
    比較適合remote sencing
    """
    contours, _ = cv2.findContours(im_in, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # n = len(contours)  # 輪廓之個數
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 100000:
            # cv_contours.append(contour)
            cv2.drawContours(im_in, [contour], 0, (255, 255, 255), -1)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
        else:
            continue
            
    im_out = cv2.fillPoly(im_in, cv_contours, (255, 255, 255))
    return im_out


def remove_small_points(src, threshold_area):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8, ltype=None)
    img = np.zeros((src.shape[0], src.shape[1]), np.uint8)  
    for i in range(1, num_labels):
        mask = labels == i             
        if stats[i][4] > threshold_area:         
            img[mask] = 255
                      
        else:
            img[mask] = 0
           
    return img


# def remove_small_points(binary_img, threshold_area):
#     """
#     消除孤立點
#     args:
#         threshold_area: 大於保留，小於去除
#     return:
#         resMatrix: 消除孤立點的圖
#     """
#     # 輸出二值化影像中 所有的連通區域
#     img_label, num = label(binary_img, connectivity=1, background=0, return_num=True) #connectivity=1--4  connectivity=2--8
    
#     #輸出連通區域的屬性

#     props = regionprops(img_label) 
#     ## adaptive threshold
#     # props_area_list = sorted([props[i].area for i in range(len(props))]) 
#     # threshold_area = props_area_list[-2]
#     resMatrix = np.zeros(img_label.shape).astype(np.uint8)
#     for i in range(0, len(props)):
#         # print('--',props[i].area)
#         if props[i].area > threshold_area:
#             tmp = (img_label == i + 1).astype(np.uint8)
            
#             #組合所有符合條件的連通區域
#             resMatrix += tmp 
#     resMatrix *= 255
    
#     return resMatrix


def canny_process(img):
    """
    Cannt
    """
    
    # 先將圖片轉成灰階
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #設kernel給dilate and erosion
    kernel = np.ones((3,3),np.uint8) 
    

    #使用canny來找線條，因為陸地的線條相較於海面會很明顯
    #找出來之線條使用膨脹來加強
    low_threshold = 1
    high_threshold = 50
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)
    dilate = cv2.dilate(edges,kernel,iterations = 3)
    edges2 = fill(dilate)

    opening1 = cv2.morphologyEx(edges2, cv2.MORPH_OPEN, kernel, iterations=2)
    res_img1 = fill(opening1)
    closing1 = cv2.morphologyEx(res_img1, cv2.MORPH_CLOSE, kernel, iterations=2)


    
    #去除小碎塊面積
    #做erosion來變回原本的形狀
    # remove_small = remove_small_points(closing1, 10000)
    # kernel = np.ones((3,3),np.uint8) 
    erosion = cv2.erode(closing1,kernel,iterations = 3) #25
    
    

    #設一個新的kernel，5*5可以有更明顯的效果，要填滿沒被填滿的區域
    #因為使用傳統的填滿，會導致整張圖變成一個顏色，因為遙感影像太多碎塊
    kernel = np.ones((5,5),np.uint8)
    opening1 = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel, iterations=10)
    fill_img1 = fill(opening1)
    remove_small2 = remove_small_points(fill_img1, 100*100)
    
    return remove_small2

def standard_process(image):
    """
    計算圖片標準差
    """
    # 讀取影像
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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
    fil = fill(result_normalized)
    # result_normalized = remove_small_points(fil, 70*70)
    
    kernel = np.ones((3,3),np.uint8)
    opening1 = cv2.morphologyEx(fil, cv2.MORPH_OPEN, kernel, iterations=2)
    closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel, iterations=2)
    fill_img1 = fill(closing1)
    remove_small = remove_small_points(fill_img1, 50*50)
    return remove_small

def otsu_process(img):
    """
    對圖片進行OTSU來二值化
    """
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret1, th1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fil = fill(th1)
    res_img1 = remove_small_points(fil, 70*70)
    
    kernel = np.ones((3,3),np.uint8)
    opening1 = cv2.morphologyEx(res_img1, cv2.MORPH_OPEN, kernel, iterations=2)
    closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel, iterations=2)
    fill_img1 = fill(closing1)
    
    # plt.subplot(211), plt.imshow(img), plt.title('RGB')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(212), plt.imshow(th1, cmap='gray'), plt.title('Otsu')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    
    return fill_img1


def hsv_process(img):
    max_value, min_value, max_pixel_values = hsv(img)
    max_pixel_values = max_pixel_values[0]
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print(max_value, min_value, max_pixel_values)
    # print('max_value: ',max_value)
    # print('min_value: ',min_value)
    
    
    h_channel = hsv_image[:, :, 0]
    threshold_value = (max_value + min_value) / 2
    diff = abs(max_value - min_value)
    # print('threshold_value: ',threshold_value)
    
    if diff <= 20:
        binary_image = np.zeros_like(h_channel)
    else:
        if 0 < max_pixel_values < 70:
            # _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
            # binary_image = cv2.bitwise_not(binary_image)
            if 70 <= max_value < 120:
                
                _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
                binary_image = cv2.bitwise_not(binary_image)
            elif 120 <= max_value <= 180:
                _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
            else:
                binary_image = np.zeros_like(h_channel)
                
                
        elif 70 <= max_pixel_values <= 180:
            if 70 <= max_value < 120:
                _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
                binary_image = cv2.bitwise_not(binary_image)
            elif 120 <= max_value <= 180:
                _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
            else:
                binary_image = np.zeros_like(h_channel)
        else:
            binary_image = np.zeros_like(h_channel)
    
    nonzero_count = np.count_nonzero(binary_image)
    total_pixels = img.shape[0] * img.shape[1]
    nonzero_ratio = nonzero_count / total_pixels
    # print(nonzero_ratio)
    
    if nonzero_ratio > 0.9:
        binary_image = np.zeros_like(h_channel)
    
    
    # plt.subplot(221), plt.imshow(img), plt.title('RGB')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(222), plt.imshow(hsv_image), plt.title('hsv_image')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(223), plt.imshow(binary_image, cmap='gray'), plt.title('binary_image')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    
    return binary_image, abs(max_value-min_value)

def hist(img):
    """
    輸入:圖片
    
    目的:計算圖片中的RGB三個通道之出現頻率最高之pixel value、
        其最高之頻率值以及pixel value是0~51所出現在圖片的占比
        
        
    """
    
    max_pixel_values = []  
    max_pixel_frequencies = []  
    pixel_frequencies = []
    
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = ['blue','springgreen','red'] 
    for i in [0,1,2]:
        #計算三個通道pixel values值方圖
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        hist = cv2.calcHist([RGB_img], [i], mask, [256], [0.0, 255.0])
        # hist = cv2.calcHist([img],[i], None, [256], [0.0,255.0]) 
        
        #最高頻率之pixel value
        max_pixel_value = np.argmax(hist)
        max_pixel_frequency = np.max(hist)
        
        #最高頻率次數
        max_pixel_values.append(max_pixel_value)
        max_pixel_frequencies.append(max_pixel_frequency)
        
        #0~50的pixel頻率出現比值
        pixel_frequency = np.sum(hist[1:50]) / np.sum(hist)
        pixel_frequencies.append(pixel_frequency)
        
    #     plt.subplot(121), plt.plot(hist, color[i])
    #     plt.title('Histrogram of Color image')

    # plt.subplot(122), plt.imshow(RGB_img), plt.title('res_img1')
    # plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    
    
    # for i, channel in enumerate(color):
    #     print(f"Max pixel value in {channel} channel: {max_pixel_values[i]}")
    #     print(f"Frequency of max pixel value in {channel} channel: {max_pixel_frequencies[i]}")
    #     print(f"Frequency of pixels in {channel} channel between 0 and 50: {pixel_frequencies[i]}")
    
    # plt.show()
    return max_pixel_values, max_pixel_frequencies, pixel_frequencies



def hsv(img):
    """
    判斷HSV中的H通道範圍
    """
    max_pixel_values = []  
    
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 提取H通道
    h_channel = hsv_image[:, :, 0]

    
    # 計算H通道非0的值
    mask = np.where(h_channel == 0, 0, 255).astype(np.uint8)
    hist = cv2.calcHist([h_channel], [0], mask, [256], [0, 256])
    max_pixel_value = np.argmax(hist)
    max_pixel_values.append(max_pixel_value)
    
    # 找頻率超過某數的值
    high_freq_values = np.where(hist > 1000)[0]
    
    if len(high_freq_values) == 0:
        print("No pixel values with frequency greater than 1000.")
        # 找到值方中的最大值和最小值
        max_value = np.max(hist)
        min_value = np.min(hist)
        
    else:
        # 在高頻率像素值中找到最大值和最小值
        max_value = None
        min_value = None

        max_value = np.max(high_freq_values)
        min_value = np.min(high_freq_values)
        
        max_value = max_value if max_value is not None else min_value
        min_value = min_value if min_value is not None else max_value

    
    #     print("Max Value:", max_value)
    #     print("Min Value:", min_value)
    


    
    return max_value, min_value, max_pixel_values

def final_process(img):
    '''
    最終影像mask合併
    '''
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    hsv, color_range = hsv_process(img)
    
    standard = standard_process(img) #標準差
    canny = canny_process(img) #線條
    output1 = cv2.add(standard, canny) #合併
    
    kernel = np.ones((3,3),np.uint8) 
    
    #開, 填滿, 閉
    opening1 = cv2.morphologyEx(output1, cv2.MORPH_OPEN, kernel, iterations=5)
    fill_img1 = fill(opening1)
    closing1 = cv2.morphologyEx(fill_img1, cv2.MORPH_CLOSE, kernel, iterations=25)
    
    
    closing2 = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel, iterations=5)
    opening2 = cv2.morphologyEx(closing2, cv2.MORPH_OPEN, kernel, iterations=10)
    fill_img2 = fill(opening2)
    fill_img2 = cv2.erode(fill_img2,kernel,iterations = 5) #25



    nonzero_count1 = np.count_nonzero(closing1)
    nonzero_count2 = np.count_nonzero(fill_img2)
    
    total_pixels = img.shape[0] * img.shape[1]
    nonzero_ratio1 = nonzero_count1 / total_pixels
    nonzero_ratio2 = nonzero_count2 / total_pixels
    # print(nonzero_ratio1,nonzero_ratio2)
    
    nonzero_diff = abs(nonzero_ratio1 - nonzero_ratio2)
    if nonzero_diff >= 0.15:
        if nonzero_ratio2 < 0.1:
            output_final = closing1
        elif nonzero_ratio1 < 0.1:
            output_final = fill_img2
        elif nonzero_ratio1 > nonzero_ratio2:
            output_final = fill_img2
        elif nonzero_ratio1 < nonzero_ratio2:
            output_final = closing1
    
    elif nonzero_ratio1 > 0.8:
        output_final = fill_img2
        
    elif nonzero_ratio2 > 0.85:
        output_final = closing1
        
    else:
        output_final = cv2.add(closing1, fill_img2) #合併
        
    
    
    
    output_final = remove_small_points(output_final, 70*70)
    # output_final = cv2.bitwise_and(closing1, closing2)
    # output_final = fill(output_final)
    #陸地mask通常都會是白色，但需要保留的是海，需要做not
    output = cv2.bitwise_not(output_final)
    
    
    # plt.subplot(221), plt.imshow(RGB_img), plt.title('RGB_img')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(222), plt.imshow(closing1, cmap='gray'), plt.title('closing1')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(223), plt.imshow(fill_img2, cmap='gray'), plt.title('fill_img2')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(224), plt.imshow(output_final, cmap='gray'), plt.title('output_final')
    # plt.xticks([]), plt.yticks([])

    # plt.tight_layout()
    # plt.show()

    
    return output, color_range

def imgProcess(filePath):
    #只要是以下副檔名都可以被讀取
    extensions = tuple(['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo'])
    for filename in os.listdir(filePath):
        
        if filename.endswith(extensions):
            # print(filename)
            dir_path = os.path.dirname(filePath) + "/sealand"
            
            if os.path.exists(dir_path) == False:
                os.makedirs(dir_path)
            
            img = cv2.imread(filePath + "/" + filename)
            
            #利用H通道來判斷是否為海洋
            # color_range = hsv(img)
            
            #計算值方圖，來判斷是否有包含陸地
            max_pixel_values, max_pixel_frequencies, pixel_frequencies = hist(img)
            
            #blue, green, red
            BMP = max_pixel_values[0]
            GMP = max_pixel_values[1]
            RMP = max_pixel_values[2]
            RMF = max_pixel_frequencies[2]
            RF = pixel_frequencies[2]
            
            output, color_range = final_process(img)
            # output, color_range = tfinal_process(img)
            
            zero_pixels = np.count_nonzero(output == 0)
            total_pixels = img.shape[0] * img.shape[1]
            zero_ratio = zero_pixels / total_pixels
            
            if cv2.countNonZero(output) == 0:
                output = cv2.bitwise_not(output)
                
                
            # final = cv2.bitwise_and(img, img, mask=output)
            # cv2.imwrite(dir_path + "/" + filename, final)
            
            if color_range >= 20:
                if  zero_ratio >= 0.9:
                    # print("Sea: ", color_range)
                    cv2.imwrite(dir_path + "/" + filename, img)
                
                else:
                    final = cv2.bitwise_and(img, img, mask=output)
                    cv2.imwrite(dir_path + "/" + filename, final)
                # print(color_range)
                #如果pixel value最大值出現在50以下
                #且紅色大於藍綠
                # cv2.imwrite(dir_path + "/hsv/" + filename, img)
                
                # if  zero_ratio >= 0.9:
                #     # print("Sea: ", color_range)
                #     cv2.imwrite(dir_path + "/" + filename, img)
                
                # else:
                #     final = cv2.bitwise_and(img, img, mask=output)
                #     cv2.imwrite(dir_path + "/" + filename, final)
                    
                # if RMP <= 50 and GMP <= 50 and BMP <= 50 and RMP >= GMP and RMP >= BMP:
                #     # cv2.imwrite(dir_path + "/50land/" + filename, img)
                #     # print("Land: ", color_range)
                #     if  zero_ratio >= 0.9:
                #         # print("Sea: ", color_range)
                #         cv2.imwrite(dir_path + "/" + filename, img)
                    
                #     else:
                #         final = cv2.bitwise_and(img, img, mask=output)
                #         cv2.imwrite(dir_path + "/" + filename, final)

                #如果紅色出現在0~50的頻率占三分之一以下
                #就判定為陸地
                # if (RF < 0.66):
                #     # cv2.imwrite(dir_path + "/66land/" + filename, img)
                #     if  zero_ratio >= 0.9:
                #         # print("Sea: ", color_range)
                #         cv2.imwrite(dir_path + "/" + filename, img)
                    
                #     else:
                #         final = cv2.bitwise_and(img, img, mask=output)
                #         cv2.imwrite(dir_path + "/" + filename, final)

                # else:
                #     # cv2.imwrite(dir_path + "/sea/" + filename, img)
                #     cv2.imwrite(dir_path + "/" + filename, img)
                    

            else:
                if  RMP <= GMP and RMP <= BMP:
                    if  zero_ratio >= 0.9:
                        # print("Sea: ", color_range)
                        cv2.imwrite(dir_path + "/" + filename, img)
                    
                    else:
                        final = cv2.bitwise_and(img, img, mask=output)
                        cv2.imwrite(dir_path + "/" + filename, final)
                else:
                    cv2.imwrite(dir_path + "/" + filename, img)
                
 
                


                
