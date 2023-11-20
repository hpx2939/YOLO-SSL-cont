# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:17:59 2023

@author: user
"""
file_path = 'D://yolov7/runs/train/aitod_4a/SPD_aitod.txt'
ssum=float(0)
average=float(0)
with open(file_path) as f:
    for line in f.read().splitlines():
        data_frac = line.split(',')
        ssum =ssum + float(data_frac[1])
        #print(data_frac[1])
        
        #print(data_frac[0])
#average = average/87
print(ssum)
print(average)