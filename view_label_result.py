#-*-coding:utf8-*-
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import cv2
import yaml
from dataset.synthetic_shapes import SyntheticShapes
from model.superedgev1 import SuperEdgeV1
from torch.utils.data import DataLoader
import os
types = ["draw_ellipses","draw_multiple_polygons","draw_star","draw_cube","draw_lines","draw_polygon",]
# types = ["draw_lines",]
for ty in types:
    # image_path = "./data/synthetic_shapes/"+ty+"/images/test/"
    # kp_path = "./data/synthetic_shapes/" + ty +"/edges/test/"

    # image_path = "./data/coco/images_v2/test2017/"
    # kp_path = "./data/coco/labels_v2/test2017/"

    image_path = "./data/coco/images_v2/val2017/"
    kp_path = "./data/coco/labels_v2/val2017/"

   

    files = os.listdir(image_path)
    files.sort()
    i = 0 
    for file_name in files:

        img = cv2.imread(image_path + file_name)
        img_pix = img.copy()
        img_obj = img.copy()
        ori_img = img.copy()

        points = np.load(kp_path + file_name[:]+".npy")
        keypoints = np.load(kp_path +"obj"+ file_name[:]+".npy")
        # print(points)
        for kp in points:
            cv2.circle(img,(int(kp[1]), int(kp[0])), radius=0, color=(0, 255, 0))
        for kp in keypoints:
            cv2.circle(img,(int(kp[1]), int(kp[0])), radius=0, color=(0, 0, 255))

        for kp in points:
            cv2.circle(img_pix,(int(kp[1]), int(kp[0])), radius=0, color=(0, 255, 0))
        for kp in keypoints:
            cv2.circle(img_obj,(int(kp[1]), int(kp[0])), radius=0, color=(0, 0, 255))
        # print(i)
        i = i+1
        
        # if(i%50 == 0):
        #     break
        #     print(i,"./result/syn_label/"+str(ty) + "/"+ str(i)+".png")

        print(img_pix.shape)
        # cv2.imwrite("./results/"+ file_name[:-4]+".png",img_pix)
        cv2.imwrite("./results/"+ file_name[:-4]+".png",img)
        # cv2.imwrite("./results/"+ file_name+"_pix.png",img_pix)
        # cv2.imwrite("./tmper/"+ty+""+ file_name+"_obj.png",img_obj)
        # cv2.imwrite("./tmper/"+ty+""+ file_name+"_oor.png",ori_img)
        # cv2.imwrite("./result/pl/coco_v1/test"+ "/"+ file_name+".png",img)

    break
print('Done')