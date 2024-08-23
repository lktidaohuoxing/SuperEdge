#-*-coding:utf8-*-
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import cv2
import yaml
from torch.utils.data import DataLoader
import sys
# from dataset.synthetic_shapes import SyntheticShapes
from model.superedgev1 import SuperEdgeV1
from model.superedge import SuperEdge
from dataset.arbitrary import COCODataset
import math
from bresenham import bresenham
from collections import deque
import torch.nn as nn
from collections import deque

from scipy.ndimage import convolve
with open('./config/visual.yaml', 'r', encoding='utf8') as fin:
    config = yaml.safe_load(fin)
from scipy.io import savemat
# 
device_ = 'cuda:1' #'cuda:2' if torch.cuda.is_available() else 'cpu'
# device = 'cpu' #'cuda:2' if torch.cuda.is_available() else 'cpu'
dataset_ = COCODataset(config['data'],config['model'], is_train=False , device=device_)
            
data_loaders = DataLoader(dataset_, batch_size=1,shuffle=False, collate_fn=dataset_.batch_collator) 

# net = MagicPoint(config['model'], device=device_)
if config['model']['name'] == 'superedge':
    net = SuperEdge(config['model'], device=device_, using_bn=config['model']['using_bn'])
elif config['model']['name'] == 'superedgev1':
    net = SuperEdgeV1(config['model'], device=device_,using_bn=config['model']['using_bn'])
net.load_state_dict(torch.load(config['model']['pretrained_model']))
# net = nn.DataParallel(net)
net.to(device_).eval()

def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img

def generate_pl_matrix(matrix,pts_kp,pts_line):
    #将线关键点全部设为1
    for kp in pts_line:
            # cv2.circle(img,(int(kp[1]), int(kp[0])), radius=0, color=(0, 255, 0))
        matrix[int(kp[0]),int(kp[1])] = 1

    for kp in pts_kp:
        if(matrix[int(kp[0]),int(kp[1])] == 0):
            matrix[int(kp[0]),int(kp[1])] = 2
        elif(matrix[int(kp[0]),int(kp[1])]  == 1):
            matrix[int(kp[0]),int(kp[1])] = 3
        # print(matrix[int(kp[0]),int(kp[1])])
    return matrix

def sigmoid(x):
    return 255 / (1 + np.exp(-40 * (x - 0.02))) + 0
    
def draw_img_with_color(img,keypoints,kp_keypoints):
    for kp in keypoints:
        colors = 2550 * float(prob[int(kp[0]),int(kp[1])] ) 

        cv2.circle(img, (int(kp[1]), int(kp[0])), radius=0, color=(0, int(colors), 0))
    for kp in kp_keypoints:
        cv2.circle(img, (int(kp[1]), int(kp[0])), radius=2, color=(0, 0, 255))
    return img
def is_surrounded_by_zeros(img, x, y):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if img[y + i, x + j] != 0:
                return False
    return True
with torch.no_grad():

    for i, data in tqdm(enumerate(data_loaders)):

        ret = net(data['raw']['img'])

        warp_img = (data['raw']['img'] * 255).cpu().numpy().squeeze().astype(np.int).astype(np.uint8)
        fuck_img = warp_img.copy()
        warp_img = cv2.merge((warp_img, warp_img, warp_img))
        prob = ret['output']['prob'].cpu().numpy().squeeze()
        keypoints = np.where(prob > 0.015)
        keypoints = np.stack(keypoints).T

        if config['model']['name'] != 'superedgev1' :
            prob_kp = ret['output_kp']['prob'].cpu().numpy().squeeze()
            kp_keypoints = np.where(prob_kp>0.015)
            kp_keypoints = np.stack(kp_keypoints).T

            img = warp_img.copy()
            img = cv2.resize(img,(int(data['raw']['pri_H'][0]),int(data['raw']['pri_W'][0])))
            dif_matrix = np.zeros((warp_img.shape[0],warp_img.shape[1]),dtype=np.uint8)
            obj_matrix = np.zeros((warp_img.shape[0],warp_img.shape[1]),dtype=np.uint8)
            for kp in keypoints:
                if( 2550 * float(prob[int(kp[0]),int(kp[1])] ) < 255 and dif_matrix[int(kp[0]),int(kp[1])] == 0 ):
                    dif_matrix[int(kp[0]),int(kp[1])] = 2550 * float(prob[int(kp[0]),int(kp[1])] ) 
                else:
                    dif_matrix[int(kp[0]),int(kp[1])] = 255

                cv2.circle(fuck_img,(int(kp[1]), int(kp[0])), radius=0, color=(0, 255, 0))

            for kp in kp_keypoints:
                if( 2550 * float(prob[int(kp[0]),int(kp[1])] ) < 255 and obj_matrix[int(kp[0]),int(kp[1])] == 0 ):
                    obj_matrix[int(kp[0]),int(kp[1])] = 2550 * float(prob[int(kp[0]),int(kp[1])] ) 
                else:
                    obj_matrix[int(kp[0]),int(kp[1])] = 255
                
            dif_matrix = image_normalization(dif_matrix) 
            obj_matrix = image_normalization(obj_matrix)      
            
            filter_size = 2  # 滤波器大小
            filter_kernel = np.ones((filter_size, filter_size)) / (filter_size**2)

            # 执行均值滤波
            dif_matrix = convolve(dif_matrix, filter_kernel)
            obj_matrix = convolve(obj_matrix, filter_kernel)

            dx = [1, -1, 0, 0, 1, 1, -1, -1]
            dy = [0, 0, 1, -1, 1, -1, 1, -1]
            visited = np.zeros_like(dif_matrix, dtype=np.uint8)
            for kp in kp_keypoints:
                x = int(kp[1])
                y = int(kp[0])
                # cv2.circle(dif_matrix, (int(y), int(kp[0])), radius=4, color=(255, int(255),0 ))
                if(visited[y,x] == 1):
                    continue
                visited[y, x] = 1 
                
                if dif_matrix[y, x] != 0:
                    queue = deque([(x, y)])
                    while queue:
                        current_x, current_y = queue.popleft()
                        # dif_matrix[current_y, current_x] = 255  # 或者你可以设置其他非零值
                        for direction in range(8):
                            new_x, new_y = current_x + dx[direction], current_y + dy[direction]
                            if 0 <= new_x < dif_matrix.shape[1] and 0 <= new_y < dif_matrix.shape[0] and dif_matrix[new_y, new_x] != 0 and not visited[new_y, new_x]:
                                queue.append((new_x, new_y))
                                visited[new_y, new_x] = 1 

            pl_img = draw_img_with_color(warp_img,keypoints,kp_keypoints)

        

            result_image = np.where(visited == 1, dif_matrix, 0)
            
            # obj_matrix = cv2.merge((obj_matrix, obj_matrix, obj_matrix))    
            # result_image = image_normalization(result_image)
            fusion_image = result_image + obj_matrix
            plus_image = dif_matrix + obj_matrix
            fusion_image = image_normalization(fusion_image)
            plus_image = image_normalization(plus_image)
            dif_matrix = cv2.merge((dif_matrix, dif_matrix, dif_matrix))  
            print(data['raw']['img_name'][0])
            cv2.imwrite("./results/"+str(data['raw']['img_name'][0])+".png",(255.-fusion_image) )
            # cv2.imwrite("./tmper/"+str(data['raw']['img_name'][2:-2])+"_final.png",(fusion_image) )
            # cv2.imwrite("./tmper/"+str(data['raw']['img_name'])+"_plus.png",(plus_image) )
            # # cv2.imwrite("./tmper/"+str(data['raw']['img_name'].item())+"_delete-.png",(result_image) )
            cv2.imwrite("./results/"+str(data['raw']['img_name'][0])+"_pix.png",(dif_matrix) )
            cv2.imwrite("./results/"+str(data['raw']['img_name'][0])+"_obg.png",(obj_matrix) )
            # cv2.imwrite("./tmper/"+str(data['raw']['img_name'].item())+"_fuck.png",fuck_img )

        
        
            # cv2.imwrite("./resss/"+str(data['raw']['img_name'].item())+".png",dif_matrix)

            #tt = erosion.copy()
            #对特定位置进行erosion
            # except:
            #     continue
        else:
            img = warp_img.copy()
            img = cv2.resize(img,(int(data['raw']['pri_H'][0]),int(data['raw']['pri_W'][0])))
            for kp in keypoints:
                cv2.circle(img,(int(kp[1]), int(kp[0])), radius=0, color=(0, 255, 0))
            cv2.imwrite("./results/"+str(data['raw']['img_name'][0])+".png",img) 

print('Done')


    