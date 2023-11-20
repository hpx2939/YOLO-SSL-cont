# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 09:10:24 2023

@author: user
"""

import utils.autoanchor as autoAnchor
 
'''
        path:  儲存yaml文件路徑，yaml文件中應包含數據集文件路徑
          n :  生成錨框的數量
   img_size :  圖片分辨率尺寸，需要將圖片縮放到img_size大小尺寸後再進行錨框計算
        thr ：  數據集中標註框寬高比最大閾值，默認是使用 超參文件 hyp.scratch.yaml 中的 “anchor_t” 參數值；默認值是 4.0；自動計算時，會自動根據你所使用的數據集，來計算合適的閾值。
        gen ：  kmean 聚類算法迭代次數，默認值是 1000
    verbose ：  打印所有結果
'''
new_anchors = autoAnchor.kmean_anchors(path=r'D:\yolov7\AITOD_ship\data.yaml', n=12, img_size=640, gen=1000, verbose=False)
print(new_anchors)
