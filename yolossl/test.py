#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 2 2023
@author: HsuPeiHsiang
usage: python test.py ../configs/testship.yaml
"""

import pandas as pd
import argparse
import shutil
import yaml
import time
import sys
import os
import cv2

import tile_ims_labels
import post_process


def delete_folder_if_exists(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        # print(f"Folder '{folder_path}' deleted.")
    else:
        pass

######################################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
######################################
# Load config and set variables
######################################

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    config_dict = yaml.safe_load(f)
    f.close()
config = dotdict(config_dict)
print("test.py config:")
print(config)



######################################
# Prepare data
######################################
t0 = time.time()

folder1 = "./data/test_imagery"
folder2 = "./results/test"
folder3 = "./runs/detect/test"

del_folder = [folder1,folder2,folder3]

for del_f in del_folder:
    print(del_f)
    delete_folder_if_exists(del_f)

######################################
# object names
######################################
# create name file
cat_int_to_name_dict = {}
namefile = os.path.join(config.yolosl_path, 'data', config.name_file_name)  #./data/data.yaml
for i, n in enumerate(config.object_names): #所有的種類
    cat_int_to_name_dict[i] = n
    if i == 0:
        os.system( 'echo {} > {}'.format(n, namefile))
    else:
        os.system( 'echo {} >> {}'.format(n, namefile))
# view
print("\nobject names ({})".format(namefile))
with open(namefile,'r') as f:
    all_lines = f.readlines()
    for l in all_lines:
        print(l)
print("cat_int_to_name_dict:", cat_int_to_name_dict)

######################################
# slice test images
######################################
new_extensions = []

if config.sliceWidth > 0:
    # # make list of test files
    print("\nslicing im_dir:", config.test_im_dir)
    print("outdir_slice_ims:", config.outdir_slice_ims)
    extensions = tuple(['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo'])
    im_list = [z for z in os.listdir(config.test_im_dir) if z.endswith(extensions)]
    # print('im_list: ', im_list)
    # os.system("pause")
    #im_list = [z for z in os.listdir(config.test_im_dir) if z.endswith(config.im_ext)]
    for im_lst in im_list:
        extension = im_lst.split(".")[-1]
        if extension not in new_extensions:
            new_extensions.append(extension)
            
    #print(new_extensions)        

    if not os.path.exists(config.outdir_slice_ims):
        os.makedirs(config.outdir_slice_ims) #, exist_ok=True)
        print("outdir_slice_ims:", config.outdir_slice_ims)
        
        # slice images
        for i,im_name in enumerate(im_list):
            # print(i,im_name)
            # os.system("pause")
            im_path = os.path.join(config.test_im_dir, im_name)
            # im_tmp = skimage.io.imread(im_path)
            im_tmp = cv2.imread(im_path,1)
            h, w = im_tmp.shape[:2]
            print(i+1, "/", len(im_list), im_name, "h, w =", h, w)
            
            if h < config.sliceHeight or w < config.sliceWidth:
                print('h < sliceHeight or w < sliceWidth')
                print("h, sliceHeight =",h,config.sliceHeight)
                print("w, sliceWidth =",w,config.sliceWidth)
                os.system("pause")
                sys.exit(0)

            # tile data
            out_name = im_name.split('.')[0]
            tile_ims_labels.slice_im_plus_boxes( #切成小chips
                im_path, out_name, config.outdir_slice_ims,
                sliceHeight=config.sliceHeight, sliceWidth=config.sliceWidth,
                overlap=config.slice_overlap, slice_sep=config.slice_sep,
                skip_highly_overlapped_tiles=config.skip_highly_overlapped_tiles,
                overwrite=config.slice_overwrite)
        im_list_test = []
        for f in sorted([z for z in os.listdir(config.outdir_slice_ims) if z.endswith(config.im_ext)]):
        #for f in sorted([z for z in os.listdir(config.outdir_slice_ims) if z.endswith(config.im_ext)]):
            im_list_test.append(os.path.join(config.outdir_slice_ims, f))
        df_tmp = pd.DataFrame({'image': im_list_test})
        df_tmp.to_csv(config.outpath_test_txt, header=False, index=False)
    else:
        print("Images already sliced to:", config.outdir_slice_ims)
        df_tmp = pd.read_csv(config.outpath_test_txt, names=['path'])
        im_list_test = list(df_tmp['path'].values)
else:
    # forego slicing
    im_list_test = []
    for f in sorted([z for z in os.listdir(config.test_im_dir) if z.endswith(config.im_ext)]):
        im_list_test.append(os.path.join(config.outdir_ims, f))
    df_tmp = pd.DataFrame({'image': im_list_test})
    df_tmp.to_csv(config.outpath_test_txt, header=False, index=False)
# print some values
print("N test images:", len(im_list))
print("N test slices:", len(df_tmp))
# view
# print("head of test files ({})".format(config.outpath_test_txt))
with open(config.outpath_test_txt,'r') as f:
    all_lines = f.readlines()
    for i,l in enumerate(all_lines):
        if i < 5:
            print(l)
        else:
            break
            
######################################
# Sea Land
# Results saved to ./data/test_imagery/images_slice/sealand
######################################
import Mulseg
# import Mulsegcom
if config.sealand == True:
    # Mulsegcom.imgProcess("./data/test_imagery/images_slice")
    Mulseg.imgProcess("./data/test_imagery/images_slice")

    dir_slice_ims = os.path.dirname(config.outdir_slice_ims) + "/sealand"
else:
    dir_slice_ims = config.outdir_slice_ims



######################################
# Yolov7
# # Results saved to runs/detect/
######################################
                
script_path = os.path.join(config.yolosl_path, 'yolov7/detect.py')

yolov7_command = 'python {} --weights {} --source {} --img {} --conf-thres {} ' \
            '--name {} --save-txt --save-conf'.format(\
            script_path, config.weights_file, dir_slice_ims,
            640, min(config.detection_threshes), 
            config.outname_infer)
                
print("\nRun yolov7:", yolov7_command)
os.system(yolov7_command)


######################################
# Post process (CPU)
######################################

pred_dir ='./runs/detect/test/labels/' #yolov7 detect result
# print('......pred_dir: ',pred_dir)
out_dir_root = os.path.join(config.yolosl_path, 'results', config.outname_infer)
# print('out_dir_root: ',out_dir_root)
os.makedirs(out_dir_root, exist_ok=True)
# print("post-proccessing:", config.outname_infer)

for detection_thresh in config.detection_threshes:

    out_csv = 'preds_refine_' + str(detection_thresh).replace('.', 'p') + '.csv'
    plot_dir = 'predict_' + str(detection_thresh).replace('.', 'p')
    if config.extract_chips:
        out_dir_chips = 'detection_chips_' + str(detection_thresh).replace('.', 'p')
    else:
        out_dir_chips = ''
    #extension = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    #for imext in extension:
    # post_process
    print("...Making output plot...")
    for imext in new_extensions:
        imext = '.'+imext
        post_process.execute(
            pred_dir=pred_dir,
            # truth_file=config.truth_file,
            raw_im_dir=config.test_im_dir,
            out_dir_root=out_dir_root,
            out_csv=out_csv,
            cat_int_to_name_dict=cat_int_to_name_dict,
            ignore_names=config.ignore_names,
            plot_dir=plot_dir,
            im_ext=imext,
            #im_ext=extension,
            out_dir_chips=out_dir_chips,
            chip_ext=imext,
            chip_rescale_frac=config.chip_rescale_frac,
            allow_nested_detections=config.allow_nested_detections,
            #max_edge_aspect_ratio=config.max_edge_aspect_ratio,
            max_edge_aspect_ratio=2.5,
            nms_overlap_thresh=0.2,
            slice_size=config.sliceWidth,
            sep=config.slice_sep,
            n_plots=config.n_plots, #raw data 有幾張圖片
            edge_buffer_test=config.edge_buffer_test,
            max_bbox_size_pix=config.max_bbox_size,
            detection_thresh=detection_thresh,
            save_json = False,
            save_label = False
            )


tf = time.time()
print("\nResults saved to: {}".format(out_dir_root))
print("\nTotal time to run inference and make plots:", tf - t0, "seconds")