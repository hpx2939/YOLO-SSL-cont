#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 2 2023
@author: HsuPeiHsiang

"""

import pandas as pd
import numpy as np
import argparse
import json
import time
import cv2
import os

# from scorer import score_one


def convert_reverse(size, box):
    '''Back out pixel coords from yolo format
    input = image_size (w,h), 
        box = [x,y,w,h]'''
    x, y, w, h = box
    dw = 1./size[0]
    dh = 1./size[1]

    w0 = w/dw
    h0 = h/dh
    xmid = x/dw
    ymid = y/dh

    x0, x1 = xmid - w0/2., xmid + w0/2.
    y0, y1 = ymid - h0/2., ymid + h0/2.

    return [x0, x1, y0, y1]


def plot_detections(im, boxes, gt_bounds=[],
               scores=[], classes=[], outfile='', plot_thresh=0.3,
               color_dict={},
               gt_color = (0, 255, 255),
               plot_line_thickness=2, show_labels=True,
               label_alpha_scale=0.85, compression_level=9,
               alpha_scaling=True, show_plots=False, skip_empty=False,
               test_box_rescale_frac=1,
               label_txt=None,
               draw_rect=True, draw_circle=False,
               verbose=False, super_verbose=False):
    """
    Plot boxes in image.
    Arguments
    ---------
    im : np.array
        Input image in array format
    boxes : np.array
        Prediction boxes with the format: [[xmin, ymin, xmax, ymax], [...] ]
    scores : np.array
        Array of prediction scores or probabilities.  If [], ignore.  If not
        [], sort boxes by probability prior to applying non-max suppression.
        Defaults to ``[]``.
    classes : np.array
        Array of object classes. Defaults to ``[]``.
    outfile : str
        Output file location, Defaults to ``''``.
    plot_thresh : float
        Minimum confidence to retain for plotting.  Defaults to ``0.5``.
    color_dict : dict
        Dictionary matching categoris to colors.
        Defaults to ``{}``.
    plot_line_thickness : int
        Thickness of bounding box lines.  Defaults to ``2``.
    show_labels : boolean
        Switch to display category labels (e.g. 'car') atop bounding boxes.
        Defaults to ``True``.
    label_alpha_scale : float
        Fraction by which to multiply alpha of label vs bounding box.
        Defaults to ``0.85``.
    compression_level : int
        Compression level of output image. Defaults to ``9`` (max compression).
    alpha_scaling : boolean
        Switch to scale bounding box opacity with confidence.
        Defaults to ``True``.
    show_plots : boolean
        Switch to display plots in real time.  Defaults to ``False``.
    skip_empty : boolean
        Switch to skip plotting if no bounding boxes. Defaults to ``False``.
    test_box_rescale_frac : float
        Value by which to rescale the output bounding box.  For example, if
        test_box_recale_frac=0.8, the height and width of the box would be
        rescaled by 0.8.  Defaults to ``1.0``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.
    """

    ##################################
    # label settings
    
    # large object labels
    font_size = 0.5
    font_width = 1  #粗度
    display_str_height = 9
    # # small object labels
    # font_size = 0.5 # 0.4 # 0.3
    # font_width = 1
    # display_str_height = 5 # 3
    # upscale plot_line_thickness
    plot_line_thickness *= test_box_rescale_frac
    font = cv2.FONT_HERSHEY_SIMPLEX
    ##################################

    if verbose:
        print("color_dict:", color_dict)
    output = im
    h, w = im.shape[:2]
    nboxes = 0
   

    # scale alpha with prob can be extremely slow since we're overlaying a
    #  a fresh image for each box, need to bin boxes and then plot. Instead,
    #  bin the scores, then plot

    # if alpha scaling, bin by scores
    if alpha_scaling:
        # if alpha scaling, bin by scores
        if verbose:
            print("Binning scores in plot_rects()...")
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html
        # define a step between 0.25 and 0.95
        bins = np.linspace(0.2, 0.95, 7)
        # bins = np.linspace(0, 1.0, 11)   # define a step of 0.1 between 0 and 1?
        # clip scores to highest value of bins
        scores_clip = np.clip(scores, 0, np.max(bins))
        # bin that each element belongs to
        inds = np.digitize(scores_clip, bins, right=True)
        unique_inds = np.sort(np.unique(inds))
        for bin_ind in unique_inds:

            # overlay for boxes and labels, respectively
            overlay = np.zeros(im.shape).astype(
                np.uint8)  # overlay = im_raw.copy()
            overlay1 = np.zeros(im.shape).astype(np.uint8)

            alpha_val = bins[bin_ind]

            boxes_bin = boxes[bin_ind == inds]
            scores_bin = scores_clip[bin_ind == inds]
            classes_bin = classes[bin_ind == inds]

            if verbose:
                print("bin_ind:", bin_ind)
                print("alpha_val:", alpha_val)
                print("scores_bin.shape:", scores_bin.shape)

            alpha = alpha_val

            # for labels, if desired, make labels a bit dimmer
            alpha_prime = max(min(bins), label_alpha_scale * alpha)
            # add boxes
            for box, score, classy in zip(boxes_bin, scores_bin, classes_bin):

                if score >= plot_thresh:
                    nboxes += 1
                    [xmin, ymin, xmax, ymax] = box
                    # [ymin, xmin, ymax, xmax] = box  # orig from github, don't know why we reversed order!...
                    left, right, top, bottom = xmin, xmax, ymin, ymax

                    # check boxes
                    if (left < 0) or (right > (w-1)) or (top < 0) or (bottom > (h-1)):
                        print("box coords out of bounds...")
                        print("  im.shape:", im.shape)
                        print("  left, right, top, bottom:",
                              left, right, top, bottom)
                        return

                    if (right < left) or (bottom < top):
                        print("box coords reversed?...")
                        print("  im.shape:", im.shape)
                        print("  left, right, top, bottom:",
                              left, right, top, bottom)
                        return

                    # get label and color
                    classy_str = str(classy) + ': ' + \
                        str(int(100*float(score))) + '%'
                    color = color_dict[classy]

                    if super_verbose:
                        #print ("  box:", box)
                        print("  left, right, top, bottom:",
                              left, right, top, bottom)
                        print("   classs:", classy)
                        print("   score:", score)
                        print("   classy_str:", classy_str)
                        print("   color:", color)

                    # add rectangle to overlay
                    if draw_rect:
                        cv2.rectangle(
                        overlay, (int(left), int(bottom)),
                        (int(right), int(top)), color,
                        plot_line_thickness,
                        lineType=1)  # cv2.CV_AA)
                    if draw_circle:
                        d = max(abs(left-right), abs(top-bottom))
                        r = int(d/2.0)
                        cx, cy = int((left+right)/2.0), int((top+bottom)/2.0)
                        cv2.circle(overlay, (cx, cy), r, color, plot_line_thickness, lineType=2)


                    # plot categories too?
                    if show_labels:
                        # adapted from visuatlizion_utils.py
                        # get location
                        display_str = classy_str  # or classy, whch is '1 = airplane'
                        # If the total height of the display strings added to the top of the bounding
                        # box exceeds the top of the image, stack the strings below the bounding box
                        # instead of above.
                        #display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
                        # Each display_str has a top and bottom margin of 0.05x.
                        total_display_str_height = (
                            1 + 2 * 0.05) * display_str_height
                        if top > total_display_str_height:
                            text_bottom = top
                        else:
                            text_bottom = bottom + total_display_str_height
                        # Reverse list and print from bottom to top.
                        (text_width, text_height), _ = cv2.getTextSize(
                            display_str, fontFace=font, fontScale=font_size,
                            thickness=font_width)  # 5, 5#font.getsize(display_str)
                        margin = np.ceil(0.1 * text_height)

                        # get rect and text coords,
                        rect_top_left = (int(left - (plot_line_thickness - 1) * margin),
                                         int(text_bottom - text_height - (plot_line_thickness + 3) * margin))
                        rect_bottom_right = (int(left + text_width + margin),
                                             int(text_bottom - (plot_line_thickness * margin)))
                        text_loc = (int(left + margin),
                                    int(text_bottom - (plot_line_thickness + 2) * margin))

                        # plot
                        # if desired, make labels a bit dimmer
                        if draw_rect:
                            cv2.rectangle(overlay1, rect_top_left, rect_bottom_right,
                                      color, -1)
                        cv2.putText(overlay1, display_str, text_loc,
                                    font, font_size, (0, 0, 0), font_width,
                                    # cv2.CV_AA)
                                    cv2.LINE_AA)

            # for the bin, combine overlay and original image
            overlay_alpha = (alpha * overlay).astype(np.uint8)
            if verbose:
                print("overlay.shape:", overlay.shape)
                print("overlay_alpha.shape:", overlay_alpha.shape)
                print("overlay.dtype:", overlay.dtype)
                print("min, max, overlay", np.min(overlay), np.max(overlay))
                #print ("output.shape:", output.shape)
                #print ("output.dtype:", output.dtype)
            # simply sum the two channels?
            # Reduce the output image where the overaly is non-
            # to use masks, see https://docs.opencv.org/3.1.0/d0/d86/tutorial_py_image_arithmetics.html
            overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            yup = np.nonzero(overlay_gray)
            output_tmp = output.astype(float)
            output_tmp[yup] *= (1.0 - alpha)
            output = cv2.add(output_tmp.astype(np.uint8), overlay_alpha)

            # add labels, if desired
            if show_labels:
                overlay_alpha1 = (alpha_prime * overlay1).astype(np.uint8)
                overlay_gray1 = cv2.cvtColor(overlay1, cv2.COLOR_BGR2GRAY)
                yup = np.nonzero(overlay_gray1)
                output_tmp = output.astype(float)
                output_tmp[yup] *= (1.0 - alpha_prime)
                output = cv2.add(output_tmp.astype(np.uint8), overlay_alpha1)

    # no alpha scaling, much simpler to plot
    else:
        
        for box, score, classy in zip(boxes, scores, classes):
            
            
            if score >= plot_thresh:
                nboxes += 1
                [xmin, ymin, xmax, ymax] = box
                if xmin == 0 and ymin ==0 and xmax == 0 and ymax == 0:
                    continue
                # [ymin, xmin, ymax, xmax] = box
                left, right, top, bottom = xmin, xmax, ymin, ymax

                # get label and color
                classy_str = str(classy) + ': ' + \
                    str(int(100*float(score))) + '%'
                color = color_dict[classy]

                if verbose:
                    #print ("  box:", box)
                    print("  left, right, top, bottom:",
                          left, right, top, bottom)
                    print("   classs:", classy)
                    print("   score:", score)

                # add rectangle
                if draw_rect:
                    
                    cv2.rectangle(output, (int(left), int(bottom)), (int(right),
                                  int(top)), color,
                                  plot_line_thickness)
                if draw_circle:
                    d = max(abs(left-right), abs(top-bottom))
                    r = int(d/2.0)
                    cx, cy = int((left+right)/2.0), int((top+bottom)/2.0)
                    cv2.circle(output, (cx, cy), r, color, plot_line_thickness, lineType=2)
                                             
                # plot categories too?
                if show_labels:
                    # adapted from visuatlizion_utils.py
                    # get location
                    display_str = classy_str  # or classy, whch is '1 = airplane'
                    # If the total height of the display strings added to the top of the bounding
                    # box exceeds the top of the image, stack the strings below the bounding box
                    # instead of above.
                    #display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
                    # Each display_str has a top and bottom margin of 0.05x.
                    total_display_str_height = (
                        1 + 2 * 0.05) * display_str_height
                    if top > total_display_str_height:
                        text_bottom = top
                    else:
                        text_bottom = bottom + total_display_str_height
                    # Reverse list and print from bottom to top.
                    (text_width, text_height), _ = cv2.getTextSize(display_str,
                                                                   fontFace=font,
                                                                   fontScale=font_size,
                                                                   thickness=font_width)  # 5, 5#font.getsize(display_str)
                    margin = np.ceil(0.1 * text_height)

                    # get rect and text coords,
                    rect_top_left = (int(left - (plot_line_thickness - 1) * margin),
                                     int(text_bottom - text_height - (plot_line_thickness + 3) * margin))
                    rect_bottom_right = (int(left + text_width + margin),
                                         int(text_bottom - (plot_line_thickness * margin)))
                    text_loc = (int(left + margin),
                                int(text_bottom - (plot_line_thickness + 2) * margin))

                    # annoying notch between label box and bounding box,
                    #    caused by rounded lines, so if
                    #    alpha is high, move everything down a smidge
                    if (not alpha_scaling) or ((alpha > 0.75) and (plot_line_thickness > 1)):
                        rect_top_left = (rect_top_left[0], int(
                            rect_top_left[1] + margin))
                        rect_bottom_right = (rect_bottom_right[0], int(
                            rect_bottom_right[1] + margin))
                        text_loc = (text_loc[0], int(text_loc[1] + margin))

                    if draw_rect:
                        cv2.rectangle(output, rect_top_left, rect_bottom_right,
                                  color, -1)
                    cv2.putText(output, display_str, text_loc,
                                font, font_size, (0, 0, 0), font_width,
                                # cv2.CV_AA)
                                cv2.LINE_AA)
                                
    # plot gt if desired                           
    if len(gt_bounds) > 0:
        # print("plotting gt bounds:",)
        plot_line_thickness_gt = 1  # plot_line_thickness
        for gt_bound in gt_bounds:
            # print("gt_bound:", gt_bound)
            [gt_cat, [ymin, xmin, ymax, xmax]] = gt_bound
            # [ymin, xmin, ymax, xmax] = gt_bound
            left, right, top, bottom = xmin, xmax, ymin, ymax
            # add rectangle
            if draw_rect:
                cv2.rectangle(output, (int(left), int(bottom)), (int(right),
                                                             int(top)), gt_color,
                                                             plot_line_thickness_gt)
            if draw_circle:
                d = max(abs(left-right), abs(top-bottom))
                r = int(d/2.0)
                cx, cy = int((left+right)/2.0), int((top+bottom)/2.0)
                cv2.circle(output, (cx, cy), r, gt_color, plot_line_thickness_gt, lineType=4)

            # plot categories too (on bottom)
            if show_labels:
                # adapted from visuatlizion_utils.py
                display_str = 'gt: ' + str(gt_cat) # or classy, whch is '1 = airplane'
                # If the total height of the display strings added to the top of the bounding
                # box exceeds the top of the image, stack the strings below the bounding box
                # instead of above.
                #display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
                # Each display_str has a top and bottom margin of 0.05x.
                total_display_str_height = (
                    1 + 2 * 0.05) * display_str_height
                if 2 < 1: # top > total_display_str_height:
                    text_bottom = top
                else:
                    text_bottom = bottom + total_display_str_height
                # Reverse list and print from bottom to top.
                (text_width, text_height), _ = cv2.getTextSize(display_str,
                                                               fontFace=font,
                                                               fontScale=font_size,
                                                               thickness=font_width)  # 5, 5#font.getsize(display_str)
                margin = np.ceil(0.1 * text_height)
                # get rect and text coords,
                rect_top_left = (int(left - (plot_line_thickness - 1) * margin),
                                 int(text_bottom - text_height - (plot_line_thickness + 3) * margin))
                rect_bottom_right = (int(left + text_width + margin),
                                     int(text_bottom - (plot_line_thickness * margin)))
                text_loc = (int(left + margin),
                            int(text_bottom - (plot_line_thickness + 2) * margin))
                # annoying notch between label box and bounding box,
                #    caused by rounded lines, so if
                #    alpha is high, move everything down a smidge
                if (not alpha_scaling) or ((alpha > 0.75) and (plot_line_thickness > 1)):
                    rect_top_left = (rect_top_left[0], int(
                        rect_top_left[1] + margin))
                    rect_bottom_right = (rect_bottom_right[0], int(
                        rect_bottom_right[1] + margin))
                    text_loc = (text_loc[0], int(text_loc[1] + margin))
                # if draw_rect:
                #     cv2.rectangle(output, rect_top_left, rect_bottom_right,
                #               color, -1)
                cv2.putText(output, display_str, text_loc,
                            font, font_size, gt_color, font_width,
                            # cv2.CV_AA)
                            cv2.LINE_AA)
 
    # resize predictions, if desired
    if test_box_rescale_frac != 1:
        height, width = output.shape[:2]
        output = cv2.resize(output, (width/test_box_rescale_frac, height/test_box_rescale_frac),
                            interpolation=cv2.INTER_CUBIC)

    # add image label if desired
    if label_txt:
        text_loc_label = (10, 20)
        cv2.putText(output, label_txt, text_loc_label,
                                font, 2*font_size, (0, 0, 0), font_width,
                                # cv2.CV_AA)
                                cv2.LINE_AA)  
                                      
    if skip_empty and nboxes == 0:
        return
    else:
        if verbose:
            print("Saving plot to:", outfile)
        cv2.imwrite(outfile, output, [
                    cv2.IMWRITE_PNG_COMPRESSION, compression_level])

    if show_plots:
        # plt.show()
        cmd = 'eog ' + outfile + '&'
        os.system(cmd)

    return


# https://github.com/avanetten/simrdwn/blob/master/simrdwn/core/post_process.py
###############################################################################
def get_global_coords(row,
                      edge_buffer_test=0,
                      max_edge_aspect_ratio=4,
                      test_box_rescale_frac=1.0,
                      max_bbox_size_pix=100,
                      rotate_boxes=False):
    """
    Get global pixel coords of bounding box prediction from dataframe row.
    Arguments
    ---------
    row : pandas dataframe row
        Prediction row from SIMRDWN
        columns:Index([u'im_name', u'Prob', u'Xmin', u'Ymin',
                        u'Xmax', u'Ymax', u'Category',
                        u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY',
                        u'Upper', u'Left', u'Height', u'Width', u'Pad',
                        u'Image_Path']
    edge_buffer_test : int
        If a bounding box is within this distance from the edge, discard.
        Set edge_buffer_test < 0 keep all boxes. Defaults to ``0``.
    max_edge_aspect_ratio : int
        Maximum aspect ratio for bounding box for boxes near the window edge.
        Defaults to ``4``.
    test_box_rescale_frac : float
        Value by which to rescale the output bounding box.  For example, if
        test_box_recale_frac=0.8, the height and width of the box would be
        rescaled by 0.8.  Defaults to ``1.0``.
    max_bbox_size_pix : float
        Max size in pixels of longest dimension of bbox
        Defaults to ``100``.
    rotate_boxes : boolean
        Switch to attempt to rotate bounding boxes.  Defaults to ``False``.
    Returns
    -------
    bounds, coords : tuple
        bounds = [xmin, xmax, ymin, ymax]
        coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    """

    xmin0, xmax0 = row['Xmin'], row['Xmax']
    ymin0, ymax0 = row['Ymin'], row['Ymax']
    upper, left = row['Upper'], row['Left']
    sliceHeight, sliceWidth = row['Height'], row['Width']
    vis_w, vis_h = row['Im_Width'], row['Im_Height'] #raw data size
    pad = row['Pad']
    dx = xmax0 - xmin0
    dy = ymax0 - ymin0
    #print(xmin0,xmax0,ymin0,ymax0,upper,left,dx,dy)
    
    # skip if too large
    #if (dx > max_bbox_size_pix) \
            #or (dy > max_bbox_size_pix):
        #print('skip if too large')
        #return [], []    
    
    # 如果物體剛好被切割就直接捨去
    if edge_buffer_test > 0:
        # skip if near edge 
        if ((float(xmin0) < (edge_buffer_test+5)) or
            (float(xmax0) > (sliceWidth - edge_buffer_test-5)) or
            (float(ymin0) < (edge_buffer_test+5)) or
                (float(ymax0) > (sliceHeight - edge_buffer_test-5))):
            #print ("Too close to edge, skipping", row, "...")
            #print ("Too close to edge, skipping...")
            return [], []
        # skip if near edge and high aspect ratio (set edge_buffer_test < 0 to skip)
        elif ((float(xmin0) < edge_buffer_test) or
                (float(xmax0) > (sliceWidth - edge_buffer_test)) or
                (float(ymin0) < edge_buffer_test) or
                (float(ymax0) > (sliceHeight - edge_buffer_test))):
            # compute aspect ratio
            if ((1.*dx/dy) > max_edge_aspect_ratio) or ((1.*dy/dx) > max_edge_aspect_ratio):
                #print ("Too close to edge, and high aspect ratio, skipping", row, "...")#
                print ("Too close to edge, and high aspect ratio, skipping...")#
                #return [], []
    
    # skip high aspect ratios
    #if ((1.*dx/dy) > max_edge_aspect_ratio) or ((1.*dy/dx) > max_edge_aspect_ratio):
        #print('skip high aspect ratios')
        #return [], []

    # set min, max x and y for each box, shifted for appropriate
    #   padding
    #print('--------------------------coord-------------------------')
    
    xmin = max(0, int(round(float(xmin0)))+left - pad)
    xmax = min(vis_w - 1, int(round(float(xmax0)))+left - pad)
    ymin = max(0, int(round(float(ymin0)))+upper - pad)
    ymax = min(vis_h - 1, int(round(float(ymax0)))+upper - pad)
    #print(xmin,xmax,ymin,ymax)
    #print('--------------------------------------------------\n')
    # rescale output box size if desired, might want to do this
    #    if the training boxes were the wrong size
    if test_box_rescale_frac != 1.0:
        dl = test_box_rescale_frac
        xmid, ymid = np.mean([xmin, xmax]), np.mean([ymin, ymax])
        dx = dl*(xmax - xmin) / 2
        dy = dl*(ymax - ymin) / 2
        x0 = xmid - dx
        x1 = xmid + dx
        y0 = ymid - dy
        y1 = ymid + dy
        xmin, xmax, ymin, ymax = x0, x1, y0, y1

    # set bounds, coords
    bounds = [xmin, xmax, ymin, ymax] #four values
    coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]] #bbox 四個角座標

    # check that nothing is negative
    if np.min(bounds) < 0:
        print("ERROR: part of bounds < 0:", bounds)
        print(" Error-causing row:", row)
        print(" Returning...")
        return
    if (xmax > vis_w) or (ymax > vis_h):
        print("part of bounds > image size:", bounds)
        print(" Error-causing row:", row)
        print(" Returning...")
        return

    return bounds, coords


###############################################################################
def augment_df(df,
               testims_dir_tot='',
               slice_sizes=[416],
               slice_sep='|',
               edge_buffer_test=0,
               max_edge_aspect_ratio=5,
               test_box_rescale_frac=1.0,
               max_bbox_size_pix=100,
               rotate_boxes=False,
               verbose=False):
    """
    Add global location columns to dataframe.
    Adapted from: # https://github.com/avanetten/simrdwn/blob/master/simrdwn/core/post_process.py
    Arguments
    ---------
    df : pandas dataframe
        Prediction dataframe from SIMRDWN
        Input columns are:
            ['im_name', 'Prob','Xmin', 'Ymin', 'Xmax', 'Ymax', 'Category']
    testims_dir_tot : str
        Full path to location of testing images
    slice_sizes : list
        Window sizes.  Set to [0] if no slicing occurred.
        Defaults to ``[416]``.
    slice_sep : str
        Character used to separate outname from coordinates in the saved
        windows.  Defaults to ``|``
    edge_buffer_test : int
        If a bounding box is within this distance from the edge, discard.
        Set edge_buffer_test < 0 keep all boxes. Defaults to ``0``.
    max_edge_aspect_ratio : int
        Maximum aspect ratio for bounding box for boxes near the window edge.
        Defaults to ``5``.
    test_box_rescale_frac : float
        Value by which to rescale the output bounding box.  For example, if
        test_box_recale_frac=0.8, the height and width of the box would be
        rescaled by 0.8.  Defaults to ``1.0``.
    max_bbox_size_pix : float
        Max size in pixels of longest dimension of bbox
        Defaults to ``100``.
    rotate_boxes : boolean
        Switch to attempt to rotate bounding boxes.  Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``
    Returns
    -------
    df : pandas dataframe
        Updated dataframe with global coords
    """

    extension_list = ['.png', '.tif', '.TIF', '.TIFF', '.tiff', '.JPG',
                      '.jpg', '.JPEG', '.jpeg']
    t0 = time.time()
    #print("Augmenting dataframe of initial length:", len(df), "...")

    # extract image root?
    # df['Image_Root_Plus_XY'] = [f.split('/')[-1] for f in df['im_name']]

    # (解析出)parse out image root and location
    im_roots, im_locs = [], []
    # for j, f in enumerate(df['Image_Root_Plus_XY'].values):
    #print('---------------------------len of df[im_name].values' , len(df['im_name'].values))
    for j, im_name in enumerate(df['im_name'].values): # chip im_name
        
        #if (j % 10000) == 0:
            #print(j)

        f = im_name.split('/')[-1]
        ext = f.split('.')[-1]
        # get im_root, (if not slicing ignore slice_sep)
        if slice_sizes[0] > 0:
            im_root_tmp = f.split(slice_sep)[0]
            xy_tmp = f.split(slice_sep)[-1]
        else:
            im_root_tmp, xy_tmp = f, '0_0_0_0_0_0_0'
        if im_root_tmp == xy_tmp:
            xy_tmp = '0_0_0_0_0_0_0'
        
            
        im_locs.append(xy_tmp)

        if '.' not in im_root_tmp:
            im_roots.append(im_root_tmp + '.' + ext)
        else:
            im_roots.append(im_root_tmp)

    if verbose:
        print("im_name[:3]", df['im_name'].values[:3])
        print("im_roots[:3]", im_roots[:3])
        print("im_locs[:3]", im_locs[:3])

    df['Image_Root'] = im_roots
    df['Slice_XY'] = im_locs
    # get positions
    df['Upper'] = [float(sl.split('_')[0]) for sl in df['Slice_XY'].values]
    df['Left'] = [float(sl.split('_')[1]) for sl in df['Slice_XY'].values]
    df['Height'] = [float(sl.split('_')[2]) for sl in df['Slice_XY'].values]
    df['Width'] = [float(sl.split('_')[3]) for sl in df['Slice_XY'].values]
    df['Pad'] = [float(sl.split('_')[4].split('.')[0])
                 for sl in df['Slice_XY'].values]
    df['Im_Width'] = [float(sl.split('_')[5].split('.')[0])
                      for sl in df['Slice_XY'].values]
    df['Im_Height'] = [float(sl.split('_')[6].split('.')[0])
                       for sl in df['Slice_XY'].values]

    if verbose:
        print("  Add in global location of each row")
    # if slicing, get global location from filename
    if slice_sizes[0] > 0:
        x0l, x1l, y0l, y1l = [], [], [], []
        br, xg, yg = [], [], []
        bad_idxs = []
        for index, row in df.iterrows(): #每次都是1個row，左到右整個橫排，一次一排
            bounds, coords = get_global_coords(
                row,
                edge_buffer_test=edge_buffer_test,
                max_edge_aspect_ratio=max_edge_aspect_ratio,
                max_bbox_size_pix=max_bbox_size_pix,
                test_box_rescale_frac=test_box_rescale_frac,
                rotate_boxes=rotate_boxes)
            if len(bounds) == 0 and len(coords) == 0:
                bad_idxs.append(index)
                [xmin, xmax, ymin, ymax] = 0, 0, 0, 0
            else:
                [xmin, xmax, ymin, ymax] = bounds
                
            bbox_ratio, xgap, ygap = cal_bbox_ratio(xmax,xmin,ymax,ymin)
            br.append(bbox_ratio)
            xg.append(xgap)
            yg.append(ygap)
            x0l.append(xmin)
            x1l.append(xmax)
            y0l.append(ymin)
            y1l.append(ymax)
            
        df['Xmin_Glob'] = x0l
        df['Xmax_Glob'] = x1l
        df['Ymin_Glob'] = y0l
        df['Ymax_Glob'] = y1l
        df['BBox_Ratio'] = br
        df['Xmax-Xmin'] = xg
        df['Ymax-Ymin'] = yg
    # if not slicing, global coords are equivalent to local coords
    else:
        df['Xmin_Glob'] = df['Xmin'].values
        df['Xmax_Glob'] = df['Xmax'].values
        df['Ymin_Glob'] = df['Ymin'].values
        df['Ymax_Glob'] = df['Ymax'].values
        bad_idxs = []

    # remove bad_idxs
    #if len(bad_idxs) > 0:
        #print("removing bad idxs near junctions:", bad_idxs)
        #df = df.drop(index=bad_idxs)
        # df = df.drop(df.index[bad_idxs])

    print("Time to augment dataframe of length:", len(df), "=",
          time.time() - t0, "seconds")
    return df



###############################################################################
def refine_df(df, groupby='Image_Path',
              groupby_cat='Category',
              cats_to_ignore=[],
              use_weighted_nms=True,
              nms_overlap_thresh=0.5,  plot_thresh=0.5,
              max_detections_per_image=1000000,
              verbose=True):
    """
    Remove elements below detection threshold, and apply non-max suppression.
    Arguments
    ---------
    df : pandas dataframe
        Augmented dataframe from augment_df()
    groupby : str
        Dataframe column indicating the image name or path.
        Defaults to ``'Image_Path'``
    groupby_cat : str
        Secondadary dataframe column to group by.  Can be used to group by
        category prior to NMS if one category is much larger than another
        (e.g. airplanes vs airports).  Set to '' to ignore.
        Defaults to ``'Category'``.
    cats_to_ignore : list
        List of categories to ignore.  Defaults to ``[]``.
    use_weighted_nms : boolean
        Switch to use weighted NMS. Defaults to ``True``.
    nms_overlap_thresh : float
        Overlap threshhold for non-max suppression. Defaults to ``0.5``.
    plot_thresh : float
        Minimum confidence to retain for plotting.  Defaults to ``0.5``.
    max_detections_per_image : int
        Maximum allowable detections per group. Defaults to ``1000000``.
        (Max detections per image if groupby='Image_Path')
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``True``.
    Returns
    -------
    df_tot : pandas dataframe
        Updated dataframe low confidence detections filtered out and NMS
        applied.
    """
    if verbose:
        print("Running refine_df()...")
    t0 = time.time()

    # group by image, and plot
    group = df.groupby(groupby)
    count = 0
    # refine_dic = {}
    print_iter = 1
    df_idxs_tot = []
    for i, g in enumerate(group): #A group is a raw data image()

        img_loc_string = g[0]
        data_all_classes = g[1]
        
        # keep only desired number of detections per image?
        data_all_classes = data_all_classes.sort_values(by='prob')[:max_detections_per_image]
    
        # image_root = data_all_classes['Image_Root'].values[0]
        if (i % print_iter) == 0 and verbose:
            print("  num boxes:", len(data_all_classes))
            print(i+1, "/", len(group), "Processing image:", img_loc_string)
        # image = cv2.imread(img_loc_string, 1)
        # if verbose:
        #    print ("  image.shape:", image.shape)

        # apply a secondary filter
        # groupby category as well so that detections can be overlapping of
        # different categories (i.e.: a helicopter on a boat)
        
        if len(groupby_cat) > 0:
            #print('-------------------------groupby_cat------------------------')
            grouim_names_set2 = data_all_classes.groupby(groupby_cat)
            for j, g2 in enumerate(group):
                class_str = g2[0]

                # skip if class_str in cats_to_ignore
                if (len(cats_to_ignore) > 0) and (class_str in cats_to_ignore):
                    print("ignoring category:", class_str)
                    continue

                data = g2[1]
                df_idxs = data.index.values
                # classes_str = np.array(len(data) * [class_str])
                scores = data['prob'].values

                if (i % print_iter) == 0 and verbose:
                    print("    Category:", class_str)
                    print("    num boxes:", len(data))
                    # print ("    scores:", scores)

                xmins = data['Xmin_Glob'].values
                ymins = data['Ymin_Glob'].values
                xmaxs = data['Xmax_Glob'].values
                ymaxs = data['Ymax_Glob'].values
                

                # set legend str?
                # if len(label_map_dict.keys()) > 0:
                #    classes_str = [label_map_dict[ztmp] for ztmp in classes_int]
                #    classes_legend_str = [str(ztmp) + ' = ' + label_map_dict[ztmp] for ztmp in classes_int]
                # else:
                #    classes_str = classes_int_str
                #    classes_legend_str = classes_str

                # filter out low probs
                high_prob_idxs = np.where(scores >= plot_thresh)
                scores = scores[high_prob_idxs]
                #classes_str = classes_str[high_prob_idxs]
                xmins = xmins[high_prob_idxs]
                xmaxs = xmaxs[high_prob_idxs]
                ymins = ymins[high_prob_idxs]
                ymaxs = ymaxs[high_prob_idxs]
                df_idxs = df_idxs[high_prob_idxs]

                boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)

                if verbose:
                    print("len boxes:", len(boxes))

                ###########
                # NMS
                if nms_overlap_thresh > 0:
                    boxes_nms_input = np.stack(
                        (xmins, ymins, xmaxs, ymaxs), axis=1)
                    if use_weighted_nms:
                        probs = scores
                    else:
                        probs = []
                    # _, _, good_idxs = non_max_suppression(
                    # good_idxs = non_max_suppression(
                    #     boxes_nms_input, probs=probs,
                    #     overlapThresh=nms_overlap_thresh)
                    
                    good_idxs = Truncated_non_max_suppression(
                        boxes_nms_input, probs=probs,
                        overlapThresh=nms_overlap_thresh)

                    if verbose:
                        print("num boxes_all:", len(xmins))
                        print("num good_idxs:", len(good_idxs))
                    if len(boxes) == 0:
                        print("Error, No boxes detected!")
                    boxes = boxes[good_idxs]
                    scores = scores[good_idxs]
                    df_idxs = df_idxs[good_idxs]
                    #classes = classes_str[good_idxs]

                df_idxs_tot.extend(df_idxs)
                count += len(df_idxs)

        # no secondary filter
        else:
            data = data_all_classes.copy()
            # filter out cats__to_ignore
            if len(cats_to_ignore) > 0:
                data = data[~data['Category'].isin(cats_to_ignore)]
            df_idxs = data.index.values
            #classes_str = np.array(len(data) * [class_str])
            scores = data['prob'].values

            if (i % print_iter) == 0 and verbose:
                print("    num boxes:", len(data))
                # print ("    scores:", scores)

            xmins = data['Xmin_Glob'].values
            ymins = data['Ymin_Glob'].values
            xmaxs = data['Xmax_Glob'].values
            ymaxs = data['Ymax_Glob'].values

            # filter out low probs
            high_prob_idxs = np.where(scores >= plot_thresh)
            scores = scores[high_prob_idxs]
            #classes_str = classes_str[high_prob_idxs]
            xmins = xmins[high_prob_idxs]
            xmaxs = xmaxs[high_prob_idxs]
            ymins = ymins[high_prob_idxs]
            ymaxs = ymaxs[high_prob_idxs]
            df_idxs = df_idxs[high_prob_idxs]

            boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)

            if verbose:
                print("len boxes:", len(boxes))

            ###########
            # NMS
            if nms_overlap_thresh > 0:
                # Try nms with pyimagesearch algorighthm
                # assume boxes = [[xmin, ymin, xmax, ymax, ...
                #   might want to split by class because we could have
                #   a car inside the bounding box of a plane, for example
                boxes_nms_input = np.stack(
                    (xmins, ymins, xmaxs, ymaxs), axis=1)
                if use_weighted_nms:
                    probs = scores
                else:
                    probs = []
                # _, _, good_idxs = non_max_suppression(
                # good_idxs = non_max_suppression(
                #     boxes_nms_input, probs=probs,
                #     overlapThresh=nms_overlap_thresh)
                
                good_idxs = Truncated_non_max_suppression(
                    boxes_nms_input, probs=probs,
                    overlapThresh=nms_overlap_thresh)

                if verbose:
                    print("num boxes_all:", len(xmins))
                    print("num good_idxs:", len(good_idxs))
                boxes = boxes[good_idxs]
                scores = scores[good_idxs]
                df_idxs = df_idxs[good_idxs]
                # classes = classes_str[good_idxs]

            df_idxs_tot.extend(df_idxs)
            count += len(df_idxs)

    #print ("len df_idxs_tot:", len(df_idxs_tot))
    df_idxs_tot_final = np.unique(df_idxs_tot)
    #print ("len df_idxs_tot unique:", len(df_idxs_tot))

    # create dataframe
    if verbose:
        print("df idxs::", df.index)
        print("df_idxs_tot_final:", df_idxs_tot_final)
    df_out = df.loc[df_idxs_tot_final]

    t1 = time.time()
    print("Initial length:", len(df), "Final length:", len(df_out))
    print("Time to run refine_df():", t1-t0, "seconds")
    return df_out  # refine_dic





###############################################################################
def non_max_suppression(boxes, probs=[], overlapThresh=0.5, verbose=False):
    """
    Apply non-max suppression.
    Adapted from:
    http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    Malisiewicz et al.
    see modular_sliding_window.py, functions non_max_suppression, \
            non_max_supression_rot
    Another option:
        https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/utils.py
    Arguments
    ---------
    boxes : np.array
        Prediction boxes with the format: [[xmin, ymin, xmax, ymax], [...] ]
    probs : np.array
        Array of prediction scores or probabilities.  If [], ignore.  If not
        [], sort boxes by probability prior to applying non-max suppression.
        Defaults to ``[]``.
    overlapThresh : float
        minimum IOU overlap to retain.  Defaults to ``0.5``.
    Returns
    -------
    pick : np.array
        Array of indices to keep
    """

    if verbose:
        print("Executing non-max suppression...")
    len_init = len(boxes)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], [], []

    # boxes_tot = boxes  # np.asarray(boxes)
    boxes = np.asarray([b[:4] for b in boxes])
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort the boxes by the bottom-right y-coordinate of the bounding box
    if len(probs) == 0:
        idxs = np.argsort(y2)
    # sort boxes by the highest prob (descending order)
    else:
        # idxs = np.argsort(probs)[::-1]
        idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs,
            np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    if verbose:
        print("  non-max suppression init boxes:", len_init)
        print("  non-max suppression final boxes:", len(pick))
    return pick


def Truncated_non_max_suppression(boxes, probs=[], overlapThresh=0.5, IOIt=0.7, IOOt=0.1, verbose=False):
    """
    https://www.sciencedirect.com/science/article/pii/S0924271622001976
    ---------
    boxes : np.array
        Prediction boxes with the format: [[xmin, ymin, xmax, ymax], [...] ]
    probs : np.array
        Array of prediction scores or probabilities.  If [], ignore.  If not
        [], sort boxes by probability prior to applying non-max suppression.
        Defaults to ``[]``.
    overlapThresh : float
        minimum IOU overlap to retain.  Defaults to ``0.5``.
    IoIt: float
        intersection of inside box threshold. Defaults to ``0.7``.
    IOOt: float
        intersection of outside box threshold. Defaults to ``0.1``.
    Returns
    -------
    pick : np.array
        Array of indices to keep
    """

    if verbose:
        print("Executing non-max suppression...")
    len_init = len(boxes)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], [], []

    # boxes_tot = boxes  # np.asarray(boxes)
    boxes = np.asarray([b[:4] for b in boxes])
    
    #如果是整數，轉換成float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    #遞增方式做排序 
    idxs = np.argsort(probs)


    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        # pick.append(i) # 保留每一類最高分的bboxs
        

        # 找交集 以下是numpy組 不是純量
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        
        # 以下是numpy組 不是純量
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        
        # 計算得分最高的那個邊界框和其餘邊界框的交集面積
        overlap = (w * h) / area[idxs[:last]]
        overlap_other = overlap             # 交集與其他預選框的比例
        overlap_self = (w * h) / area[i]    # 交集與本預選框的比例
        union = (area[idxs[:last]] + area[i]) - (w * h)


        # 如果選擇到面積小的預選框，有比他更大的面積，就刪除此預選框
        if np.any(overlap_self > IOIt) and np.any(overlap_other <= IOOt):
            if np.any(np.less(area[i], area[idxs[:last]])):
                idxs = np.delete(idxs,[last])
                
            else:
                pick.append(i)
                idxs = np.delete(idxs,
                    np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
                
        # 如果是選擇到最大的預選框，且分數也最高，就採用此預選框作為bbox        
        else:
            
            # 刪除與last重疊超過閥值的索引
            pick.append(i)
            idxs = np.delete(idxs,
                np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    if verbose:
        print("  non-max suppression init boxes:", len_init)
        print("  non-max suppression final boxes:", len(pick))
    return pick
###############################################################################
###############################################################################
def output_label(outfile, df_filt):
    """
    by Hsu
    output path: .\results\test\labels
    
    
    to_csv: 1 575 60 582 68 0.228149
    """
    
    mul_data= []
    output_data = []
    print("Outfile: ",outfile)
    xmin = df_filt['Xmin_Glob'].values
    ymin = df_filt['Ymin_Glob'].values
    xmax = df_filt['Xmax_Glob'].values
    ymax = df_filt['Ymax_Glob'].values
    probs = df_filt['prob'].values
    cat_int = df_filt['cat_int'].values
    
    
    if len(xmin) > 1:
        for i in range(len(xmin)):
            
            if int(xmin[i])==0 and int(ymin[i])==0 and int(xmax[i])==0 and int(ymax[i])==0:
                continue
   
            mul_data= []
            mul_data.append(str(int(cat_int[i])))
            mul_data.append(str(int(xmin[i])))
            mul_data.append(str(int(ymin[i])))
            
            mul_data.append(str(int(xmax[i])))
            mul_data.append(str(int(ymax[i])))
            mul_data.append(str(float(probs[i])))
            output_data.append(mul_data)
            
    elif len(xmin) == 1: 

        mul_data= []
        mul_data.append(str(int(cat_int)))
        mul_data.append(str(int(xmin)))
        mul_data.append(str(int(ymin)))
        mul_data.append(str(int(xmax)))
        mul_data.append(str(int(ymax)))
        mul_data.append(str(float(probs)))
        output_data.append(mul_data)
     
    else: 
        print("------output_label----error-------")
    
    
    output_data_new = pd.DataFrame(output_data)
    output_data_new.to_csv(outfile, header=False, index=False, sep=' ') 
    

def cal_bbox_ratio(xmax,xmin,ymax,ymin):
    x = xmax - xmin
    y = ymax - ymin
    if x == 0 or y == 0:
        return 0, x, y
    elif x >= y:
        return (x / y), x, y
    elif y > x:
        return (y / x), x, y
    

###############################################################################
###############################################################################
def execute(pred_dir='./yolov5/runs/detect/test/labels/',
            raw_im_dir='./wdata', #test data path
            im_ext='.jpg',
            ignore_names=[],
            cat_int_to_name_dict={},
            detection_thresh=0.2,
            nms_overlap_thresh=0.5,
            test_box_rescale_frac=1.0,
            max_edge_aspect_ratio=5,
            allow_nested_detections=True,
            n_plots=4,
            slice_size=416,
            max_bbox_size_pix=100,
            sep='__',
            show_labels=False,
            out_dir_root='./results/test/',
            out_csv='preds_refine_0p25.csv',
            out_dir_chips='./data/test_imagery/images_slice/',
            chip_rescale_frac=1.1,
            chip_ext='.png',
            plot_dir='preds_plot_0p25',
            groupby='image_path',
            groupby_cat='category',
            max_detections_per_image=1000000,
            edge_buffer_test=1,
            verbose=False,
            super_verbose=False,
            save_json = False,
            save_label = False
            ):
    '''
    (cls, *xywh, conf)
            
    '''
    # 輸出json檔案
    content_json = []
    
    t0 = time.time()
    
    # a few random variabls that should not need altered
    # B G R
    colors = 40*[(0, 0,255), (255, 0, 0), (0, 255, 0), (0, 140, 255),
              (0, 255, 125), (125, 125, 125)]

    # Outputs
    outpath_refined_df = os.path.join(out_dir_root, out_csv)    #0p25在result folder
    plot_dir = os.path.join(out_dir_root, plot_dir)             #0p25在result folder
    label_dir = 'labels'
    label_dir = os.path.join(out_dir_root, label_dir)
    out_dir_chips_tot = os.path.join(out_dir_root, out_dir_chips) #''

    for d in [plot_dir, out_dir_chips_tot, label_dir]:
        os.makedirs(d, exist_ok=True)

    # get list of predictions, and read data
    df_raw_list = []
    for txt_name in sorted([z for z in os.listdir(pred_dir) if z.endswith('.txt')]):  #讀取yolo detect後的txt files
        pred_txt_path = os.path.join(pred_dir, txt_name)
        #print("pred_txt_path:", pred_txt_path)
        im_name = txt_name.split('.txt')[0]  
        
        df_raw_part_frac = pd.read_csv(pred_txt_path, header=None, index_col=None, sep=' ',
                                 names=['cat_int', 'x_frac', 'y_frac', 'w_frac', 'h_frac', 'prob'])#把txt裡面的值提取
        # print("df_raw_part_frac", df_raw_part_frac)
        if len(df_raw_part_frac) > 0: #如果txt裡面有值 就會進來
            '''
            Convert each yolo coord to pixel coords, assume images have been sliced and
            # naming convention from tile_ims_labels.py is:
               out_name + slice_sep + str(y) + '_' + str(x) + '_'
               + str(sliceHeight) + '_' + str(sliceWidth)
               + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
               + '.txt')
             '''
            im_h = int(im_name.split(sep)[-1].split('_')[2])
            im_w = int(im_name.split(sep)[-1].split('_')[3])
            out_data = []
            out_cols = ['im_name', 'prob', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'cat_int', 'category']
            for data_frac in df_raw_part_frac.values:
                #print("data_frac:", data_frac)
                cat_int = int(data_frac[0]) #txt的分類  cat_int是數字
                # get the category as a string
                if cat_int_to_name_dict:
                    cat_str = cat_int_to_name_dict[cat_int] #種類(data裡面的分類名稱) 是英文不是數字
                else:
                    cat_str = ''
                #print('cat_str: ',cat_str)
                prob = float(data_frac[5]) #機率
                # get pixel coords of yolo coords
                yolosl_box = data_frac[1:5] #yolo的座標
                pix_coords_float = convert_reverse((im_w, im_h), yolosl_box)#yolo轉換座標到
                [x0, x1, y0, y1] = pix_coords_float
                # # convert to int?
                # [x0, x1, y0, y1] = [int(round(b, 2)) for b in pix_coords_float]
                out_data.append([im_name, prob, x0, y0, x1, y1, cat_int, cat_str]) 
            df_raw_part = pd.DataFrame(out_data, columns=out_cols) 
            df_raw_list.append(df_raw_part) #每張圖片的預測轉換結果
            

    # create master df
    if len(df_raw_list) == 0:#每張圖片的預測轉換結果
        print("No output files to process, returning!")
        return
    else:
        df_raw = pd.concat(df_raw_list) #把每張圖片yolo預測轉換結果，用成同一個matrix (im_name, prob, x0, y0, x1, y1, cat_int, cat_str)
        df_raw.index = range(len(df_raw))
        #print('df_raw.index:',df_raw.index)
        if verbose:
            print("df_raw.head:", df_raw.head())
                            
        

    # convert coords, then make some plots for tiled imagery
    
    # get image names without appended slice coords
    im_name_root_list = [z.split(sep)[0] for z in df_raw['im_name'].values] #txt檔名 變成list
    df_raw['im_name_root'] = im_name_root_list
    
    # filter by prob 設一個thre來過濾低的
    df_raw = df_raw[df_raw['prob'] >= detection_thresh] 
    
    # filter by ignore_names
    if ignore_names:
        if len(ignore_names) > 0:
            df_raw = df_raw[~df_raw['category'].isin(ignore_names)]
    
    # get image path 每一個圖片的path
    im_path_list = [os.path.join(raw_im_dir, im_name + im_ext) for 
                        im_name in df_raw['im_name_root'].values] 
    df_raw['image_path'] = im_path_list
    #print('\nlen of df_raw-------------------------------------',len(df_raw))
    #print(df_raw)
    # add global coords to dataset
    # Add global location columns to dataframe.(augment_df)
    df_tiled_aug = augment_df(df_raw,
                    testims_dir_tot=raw_im_dir,
                    slice_sizes=[slice_size],
                    slice_sep=sep,
                    edge_buffer_test=edge_buffer_test,
                    max_edge_aspect_ratio=max_edge_aspect_ratio,
                    test_box_rescale_frac=test_box_rescale_frac,
                    max_bbox_size_pix=max_bbox_size_pix,
                    rotate_boxes=False)  
    #print('len of df_tiled_aug-------------------------------------',len(df_tiled_aug))
    #print('..............df_tiled_aug:',df_tiled_aug)
    # print("df_tiled_aug;", df_tiled_aug)
    
    # filter out low detections?
    if allow_nested_detections:
        groupby_cat_refine = groupby_cat
    else:
        groupby_cat_refine = ''
    df_refine = refine_df(df_tiled_aug,
                                       groupby=groupby,
                                       groupby_cat=groupby_cat_refine,
                                       nms_overlap_thresh=nms_overlap_thresh,
                                       plot_thresh=detection_thresh,
                                       max_detections_per_image=max_detections_per_image,
                                       verbose=False)
    
    if verbose:
        print("df_refine.columns:", df_refine.columns)                                   
        print("df_refine.head:", df_refine.head())                                   
        print("df_refine.iloc[0]:", df_refine.iloc[0])                                   

    # save refined df
    df_refine.to_csv(outpath_refined_df)
    
    # create color_dict
    color_dict = {}
    for i,c in enumerate(sorted(np.unique(df_refine['category'].values))):
        color_dict[c] = colors[i]
    
    # plots (make sure to get outputs for all images, even if no predictions)
    if verbose:
        print("\n...plots...")
    im_names_tiled = sorted([z.split(im_ext)[0] for z in os.listdir(raw_im_dir) if z.endswith(im_ext)])
    im_names_set = set(df_refine['im_name_root'].values) #image file name(raw data not split)
    #print("im_names_set:", im_names_set)
    if len(im_names_set) == 0:
        print("No images found in", raw_im_dir, "with passed extension:", im_ext)
        print("Returning...")
        return
    # im_names_tiled = sorted(np.unique(df_refine['im_name_root']))  
    # score_agg_tile = []
    tot_detections = 0
    print('\n' + im_ext+'\n')
    for i,im_name in enumerate(im_names_tiled):

      
        # print('\n')
        print(i+1, "/", len(im_names_tiled), im_name)
        
        
        im_path = os.path.join(raw_im_dir, im_name + im_ext ) #raw(dataset test folder path), im_name(detect後裡面jpg file names), ext(副檔名jpg)
        #print(im_path)

        #create output dir
        outfile_plot_label = os.path.join(label_dir, im_name + '.txt')
        outfile_plot_image = os.path.join(plot_dir, im_name + im_ext) #plot folder
        save_label = True
        save_json = True
        
        # if no detections, write empty files
        if im_name not in im_names_set:
            print('no detections, write empty files.....')
            boxes, probs, classes, box_names = [], [], [], []
            save_label = False
            save_json = False
            

            
        # else, get all boxes for this image, create a list of box names too
        else:
            df_filt = df_refine[df_refine['im_name_root'] == im_name]
            boxes = df_filt[['Xmin_Glob', 'Ymin_Glob', 'Xmax_Glob', 'Ymax_Glob']].values
            probs = df_filt['prob']
            classes = df_filt['category']
            cat_int = df_filt['cat_int']


            tot_detections += len(boxes)
            
            print(" n boxes:", len(boxes))

            #geom_list_geo, geom_list_pix = [], []
            box_names = []
            for j, bbox in enumerate(boxes):
                prob, classs = probs.values[j], classes.values[j]

                box_name_sep = '__'
                box_name_tmp = im_name + box_name_sep + str(classs) + box_name_sep + str(np.round(prob, 3)) \
                                + box_name_sep + str(int(bbox[0])) + box_name_sep + str(int(bbox[1])) \
                                + box_name_sep + str(int(bbox[2])) + box_name_sep + str(int(bbox[3]))
                box_name_tmp = box_name_tmp.replace('.', 'p')
                box_names.append(box_name_tmp)

        # get gt boundaries, if desired
        bounds_gt = []


        label_txt = ''
                    
        # plot
        if i < n_plots:
            if verbose:
                print("Making output plot...")
            im_cv2 = cv2.imread(im_path)   #raw(dataset test folder path), im_name(detect後裡面jpg files), ext(副檔名)
            
            
            if save_label:
                output_label(outfile=outfile_plot_label , df_filt=df_filt)
                
            if save_json:
                file_name = im_name + im_ext
                xmin = df_filt['Xmin_Glob'].values
                ymin = df_filt['Ymin_Glob'].values
                xmax = df_filt['Xmax_Glob'].values
                ymax = df_filt['Ymax_Glob'].values
                probs = df_filt['prob'].values
                cat_int = df_filt['cat_int'].values

                if len(xmin) > 1:
                    for i in range(len(xmin)):
                        bbox = [str(int(xmin[i])),\
                                str(int(ymin[i])),\
                                str(int(xmax[i])),\
                                str(int(ymax[i]))]
                        if int(xmin[i])==0 and int(ymin[i])==0 and int(xmax[i])==0 and int(ymax[i])==0:
                            continue
                        content_dic = {
                            "name": file_name,
                            "category": str(int(cat_int[i])),
                            "bbox": bbox,
                            "score": str(float(probs[i]))
                        }
                        content_json.append(content_dic)
               
                        
                elif len(xmin) == 1: 
                    bbox = [str(int(xmin[0])),\
                            str(int(ymin[0])),\
                            str(int(xmax[0])),\
                            str(int(ymax[0]))]
                        
                    content_dic = {
                        "name": file_name,
                        "category": str(int(cat_int[0])),
                        "bbox": bbox,
                        "score": str(float(probs[0]))
                    }
                    content_json.append(content_dic)
                 
                else: 
                    print("------output_json----error-------")


                
            plot_detections(im_cv2, boxes, 
                   gt_bounds=bounds_gt,
                   scores=probs, 
                   outfile=outfile_plot_image, 
                   plot_thresh=detection_thresh, 
                   classes=classes, 
                   color_dict=color_dict,
                   plot_line_thickness=3, 
                   show_labels=show_labels,
                   alpha_scaling=False, label_alpha_scale=0.85, 
                   compression_level=8,
                   show_plots=False, skip_empty=False,
                   test_box_rescale_frac=1,
                   draw_circle=False, draw_rect=True,
                   label_txt=label_txt,
                   verbose=super_verbose, super_verbose=False)
                   
        #print('len(out_dir_chips): ',len(out_dir_chips))    
        
        # extract image chips
        if len(out_dir_chips) > 0:
            print("   Extracting chips around detected objects...")
            # image = skimage.io.imread(im_path) #raw test data path
            image = cv2.imread(im_path, 1)
            
                
            for bbox, box_name in zip(boxes, box_names):
                xmin0, ymin0, xmax0, ymax0 = bbox
                # adjust bounding box to be slightly larger if desired
                # rescale output box size if desired, might want to do this
                #    if the training boxes were the wrong size
                if chip_rescale_frac != 1.0:
                    dl = chip_rescale_frac
                    xmid, ymid = np.mean([xmin0, xmax0]), np.mean([ymin0, ymax0])
                    dx = dl*(xmax0 - xmin0) / 2
                    dy = dl*(ymax0 - ymin0) / 2
                    xmin = max(0, int(np.rint(xmid - dx)))
                    xmax = int(np.rint(xmid + dx))
                    ymin = max(0, int(np.rint(ymid - dy)))
                    ymax = int(np.rint(ymid + dy))
                else:
                    xmin, ymin, xmax, ymax = int(xmin0), int(ymin0), int(xmax0), int(ymax0)

                outpath_chip = os.path.join(out_dir_chips_tot, box_name + chip_ext)
                if not os.path.exists(outpath_chip):
                    # extract image
                    window_c = image[ymin:ymax, xmin:xmax]
                    # skimage.io.imsave(outpath_chip, window_c, check_contrast=False)
                    cv2.imwrite(outpath_chip, window_c)
                    
    if save_json:
        # content_json = content_json.tolist()
        # print(content_json)
        with open(os.path.join(out_dir_root, 'result.json'), 'w') as f:
            json.dump(content_json, f)
        
    print("\nAnalyzed", len(im_names_set), "images, detected", tot_detections, "objects")
    print("Exection time = ", time.time() - t0, "seconds")
    return
        

###############################################################################
###############################################################################
if __name__ == "__main__":

    # Construct argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='/root/yolosl/results/',
                        help="prediction location")
    parser.add_argument('--raw_im_dir', type=str, default='/wdata',
                        help="location of raw imagery")
    parser.add_argument('--cat_int_to_name_dict', type=dict, default={},
                        help="dictionary with keys=int and vals=strings (e.g. {0:'car})")
    parser.add_argument('--im_ext', type=str, default='.tif',
                        help="extension for images")
    parser.add_argument('--detection_thresh', type=float, default=0.16,
                        help="yolo threshold")
    parser.add_argument('--nms_overlap_thresh', type=float, default=0.5,
                        help="non max suppression, to turn of set to 0")
    parser.add_argument('--allow_nested_detections', type=int, default=0,
                        help="switch to allow one detection to exist inside another")
    parser.add_argument('--n_plots', type=int, default=20,
                        help="Num plots to make")
    parser.add_argument('--slice_size', type=int, default=416,
                        help="Slice size of raw imagery")
    parser.add_argument('--sep', type=str, default='__',
                        help="separator between image name and slice coords")
    parser.add_argument('--show_labels', type=int, default=0,
                        help="Switch to show category labels in plots (0, 1)")
    parser.add_argument('--out_dir_root', type=str, default='./results/',
                        help="output directory")
    parser.add_argument('--out_csv', type=str, default='preds_refine.csv',
                        help="output filename")
    parser.add_argument('--plot_dir', type=str, default='pred_plots',
                        help="output dir plots")
    parser.add_argument('--groupby', type=str, default='image_path',
                        help="group predictions by this string")
    parser.add_argument('--groupby_cat', type=str, default='category',
                        help="group predictions by this string")
    parser.add_argument('--out_dir_chips', type=str, default='',
                        help="output directory for extracted detection chips, set to '' to ignore")
    parser.add_argument('--chip_ext', type=str, default='.png',
                        help="extension for extracted image chips")
    parser.add_argument('--chip_rescale_frac', type=float, default=1.1,
                        help="fraction to extend detection when extracting chips")
                        
    args = parser.parse_args()

    execute(pred_dir=args.pred_dir,
            raw_im_dir=args.raw_im_dir,
            im_ext=args.im_ext,
            detection_thresh=args.detection_thresh,
            nms_overlap_thresh=args.nms_overlap_thresh,
            allow_nested_detections=bool(args.allow_nested_detections),
            n_plots=args.n_plots,
            slice_size=args.slice_size,
            sep=args.sep,
            show_labels=bool(args.show_labels),
            out_dir_root=args.out_dir_root,
            out_csv=args.out_csv,
            out_dir_chips=args.out_dir_chips,
            chip_ext=args.chip_ext,
            chip_rescale_frac=args.chip_rescale_frac,
            plot_dir=args.plot_dir,
            groupby=args.groubpy,
            groupby_cat=args.groupby_cat,
            edge_buffer_test=1,
            verbose=True,
            super_verbose=False,
            save_json = True,
            save_label = True
            )
