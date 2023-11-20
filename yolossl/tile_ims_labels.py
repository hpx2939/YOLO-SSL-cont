#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 2 2023
@author: HsuPeiHsiang

"""


import os
import cv2
import time


def slice_im_plus_boxes(image_path, out_name, out_dir_images, 
             sliceHeight=416, sliceWidth=416,
             overlap=0.1, slice_sep='|', pad=0,
             skip_highly_overlapped_tiles=False,
             overwrite=False, out_ext='.png'):

    """
    Slice a large image into smaller windows

    Arguments
    ---------
    image_path : str
        Location of image to slice
    out_name : str
        Root name of output files (coordinates will be appended to this)
    out_dir_images : str
        Output directory for images
    sliceHeight : int
        Height of each slice.  Defaults to ``416``.
    sliceWidth : int
        Width of each slice.  Defaults to ``416``.
    overlap : float
        Fractional overlap of each window (e.g. an overlap of 0.2 for a window
        of size 256 yields an overlap of 51 pixels).
        Default to ``0.1``.
    slice_sep : str
        Character used to separate outname from coordinates in the saved
        windows.  Defaults to ``|``
    out_ext : str
        Extension of saved images.  Defaults to ``.png``.

    Returns
    -------
    None
    """
    # 讀取圖片###################
    im_ext = '.' + image_path.split('.')[-1] #讀取輸入的圖片副檔名
    t0 = time.time()
    image = cv2.imread(image_path,1)
    print("image.shape:", image.shape)

    # 決定切割圖片尺寸 ##########
    win_h, win_w = image.shape[:2]
    dx = int((1. - overlap) * sliceWidth)  #移動的pixels數 x
    dy = int((1. - overlap) * sliceHeight) #移動的pixels數 y
    
    n_ims = 0 #計算幾張切割圖片
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):
            n_ims += 1

            if (n_ims % 100) == 0:
                print(n_ims)

            # make sure we don't have a tiny image on the edge
            if y0+sliceHeight > image.shape[0]:#預防slice的width or height 比原圖還大
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (y0+sliceHeight - image.shape[0]) > (0.6*sliceHeight):#overlap 不能0.6以上
                        continue
                    else:
                        y = image.shape[0] - sliceHeight
                else:
                    y = image.shape[0] - sliceHeight
            else:
                y = y0
                
            if x0+sliceWidth > image.shape[1]:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (x0+sliceWidth - image.shape[1]) > (0.6*sliceWidth):
                        continue
                    else:
                        x = image.shape[1] - sliceWidth
                else:
                    x = image.shape[1] - sliceWidth
            else:
                x = x0

            # extract image
            window_c = image[y:y + sliceHeight, x:x + sliceWidth]
            outpath = os.path.join(
                out_dir_images,
                out_name + slice_sep + str(y) + '_' + str(x) + '_'
                + str(sliceHeight) + '_' + str(sliceWidth)
                + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                + im_ext)
            if not os.path.exists(outpath):
                # skimage.io.imsave(outpath, window_c, check_contrast=False)
                cv2.imwrite(outpath, window_c)
            elif overwrite:
                # skimage.io.imsave(outpath, window_c, check_contrast=False)
                cv2.imwrite(outpath, window_c)
            #else:
                #print("outpath {} exists, skipping".format(outpath))
                                                                                                 
    print("Num slices:", n_ims,
          "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
    print("Time to slice", image_path, time.time()-t0, "seconds")
    
    return
    
