#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:22:36 2021

@author: avanetten

Script to execute yoltv5 testing

python testcont.py ../configs/testship.yaml
"""

import tile_ims_labels
import post_process

import pandas as pd
import argparse
import shutil
import yaml
import time
import sys
import cv2
import os


def delete_folder_if_exists(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        # print(f"Folder '{folder_path}' deleted.")
    else:
        pass

def copyfile2newfolder(source_folder, destination_folder):


    os.makedirs(destination_folder, exist_ok=True)   
    file_list = os.listdir(source_folder)
    
    for file_name in file_list:
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder, file_name)
        shutil.copy2(source_file_path, destination_file_path)

def clearfileinfloder(folder_path):
    for root, dirs, files in os.walk(folder_path):
    
          for file in files:
              file_path = os.path.join(root, file)
              os.remove(file_path)
          
          for dir in dirs:
              dir_path = os.path.join(root, dir)
              shutil.rmtree(dir_path)

def check_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0

######################################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
######################################
# Load config and set variables
######################################
clearfileinfloder('./sealand')
parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    config_dict = yaml.safe_load(f)
    f.close()
config = dotdict(config_dict)
print("test.py config:")
print(config)

yolt_src_path = os.path.join(config.yolosl_path, 'yoltv5')
print("yoltv5_execute_test.py: yolt_src_path:", yolt_src_path)
sys.path.append(yolt_src_path)


######################################
# Detele the old folder
######################################
t0 = time.time()

folder1 = "./data/test_imagery"
folder2 = "./results/test"
folder3 = "./runs/detect/test"

del_folder = [folder1,folder2]

for del_f in del_folder:
    print(del_f)
    delete_folder_if_exists(del_f)

######################################
# create name file
######################################

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
            im_tmp = cv2.imread(im_path)
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
print("head of test files ({})".format(config.outpath_test_txt))
with open(config.outpath_test_txt,'r') as f:
    all_lines = f.readlines()
    for i,l in enumerate(all_lines):
        if i < 5:
            print(l)
        else:
            break
            
######################################
# Sea Land
# Results saved to ./sealand
######################################
import Mulseg
# import Mulsegcom
# dir_slice_ims = os.path.dirname(config.outdir_slice_ims) + "/sealand"
dir_slice_ims = './sealand'
if os.path.exists(dir_slice_ims) == False:
    os.makedirs(dir_slice_ims)
    
if config.sealand == True:
    # Mulsegcom.imgProcess("D://yoltv5/yoltv5/data/test_imagery/images_slice")
    Mulseg.imgProcess("./data/test_imagery/images_slice")
else:
    copyfile2newfolder(config.outdir_slice_ims, dir_slice_ims)


######################################
# Yolov7
# # Results saved to runs/detect/
######################################

# script_path = os.path.join(config.yolosl_path, 'yolov7/detect.py')

# yolt_cmd = 'python {} --weights {} --source {} --img {} --conf-thres {} ' \
#             '--name {} --save-txt --save-conf --device 0'.format(\
#             script_path, config.weights_file, dir_slice_ims,
#             640, min(config.detection_threshes), 
#             config.outname_infer)
                
######################################
# 判定是否繼續執行合併
######################################
while True:
    if check_folder_empty(dir_slice_ims):
        print("Folder is empty. Combining the split images.")
        break
    time.sleep(1)

######################################
# Post process (Combine the split images)
######################################

pred_dir ='./runs/detect/test/labels/' #yolov7 detect result
out_dir_root = os.path.join(config.yolosl_path, 'results', config.outname_infer)
os.makedirs(out_dir_root, exist_ok=True)

# print('......pred_dir: ',pred_dir)
# print('out_dir_root: ',out_dir_root)
# print("post-proccessing:", config.outname_infer)

print("Waiting for detection !!")

# print(dir_slice_ims)
# print(dir_slice_ims_len)

# while True:
#     dir_slice_ims_len = len(os.listdir(dir_slice_ims))
#     if dir_slice_ims_len == 0:
#         break

print("---------------Start merging images---------------")

for detection_thresh in config.detection_threshes:

    out_csv = 'preds_refine_' + str(detection_thresh).replace('.', 'p') + '.csv'
    plot_dir = 'predict_' + str(detection_thresh).replace('.', 'p')
    if config.extract_chips:
        out_dir_chips = 'detection_chips_' + str(detection_thresh).replace('.', 'p')
    else:
        out_dir_chips = ''
    #extension = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    # print("...Making output plot...")
    for imext in new_extensions:
        imext = '.'+imext
        post_process.execute(
            pred_dir=pred_dir,
            raw_im_dir=config.test_im_dir,
            out_dir_root=out_dir_root,
            out_csv=out_csv,
            cat_int_to_name_dict=cat_int_to_name_dict,
            ignore_names=config.ignore_names,
            plot_dir=plot_dir,
            im_ext=imext,
            out_dir_chips=out_dir_chips,
            chip_ext=imext,
            chip_rescale_frac=config.chip_rescale_frac,
            allow_nested_detections=config.allow_nested_detections,
            max_edge_aspect_ratio=2.5,
            nms_overlap_thresh=0.2,
            slice_size=config.sliceWidth,
            sep=config.slice_sep,
            n_plots=config.n_plots, #raw data 有幾張圖片
            edge_buffer_test=config.edge_buffer_test,
            max_bbox_size_pix=config.max_bbox_size,
            detection_thresh=detection_thresh)


clearfileinfloder(folder3)
labels_folder_path = os.path.join(folder3, 'labels')
os.makedirs(labels_folder_path, exist_ok=True)
tf = time.time()
print("\nResults saved to: {}".format(out_dir_root))
print("\nTotal time to run inference and make plots:", tf - t0, "seconds")