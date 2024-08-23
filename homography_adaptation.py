import yaml
import os
import torch
from tqdm import tqdm
from math import pi
import kornia
import cv2
import numpy as np
from utils.params import dict_update
from solver.nms import box_nms
from utils.tensor_op import erosion2d
from dataset.utils.homographic_augmentation import sample_homography,ratio_preserving_resize
from model.superedgev1 import SuperEdgeV1
from model.superedge import SuperEdge
import math
import random
from bresenham import bresenham
import time
import argparse

homography_adaptation_default_config = {
        'num': 50,
        'aggregation': 'max',
        'valid_border_margin': 3,
        'homographies': {
            'translation': True,
            'rotation': True,
            'scaling': True,
            'perspective': True,
            'scaling_amplitude': 0.1,
            'perspective_amplitude_x': 0.1,
            'perspective_amplitude_y': 0.1,
            'patch_ratio': 0.5,
            'max_angle': pi,
        },
        'filter_counts': 0
}


def read_image(img_path):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
    Returns
      grayim: grayscale image
    """
    grayim = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if grayim is None:
        raise Exception('Error reading image %s' % img_path)
    return grayim

def to_tensor(image, device):
    H,W = image.shape
    image = image.astype('float32') / 255.
    image = image.reshape(1, H, W)
    image = torch.from_numpy(image).view(1,1,H,W).to(device)
    return image

def one_adaptation(net, raw_image, probs, probs_kp, counts, images, config, device='cpu',model_name='superedgev1'):
    """
    :param probs:[B,1,H,W]
    :param counts: [B,1,H,W]
    :param images: [B,1,H,W,N]
    :return:
    """
    B, C, H, W, _ = images.shape
    #sample image patch
    M = sample_homography(shape=[H, W], config=config['homographies'],device=device)
    M_inv = torch.inverse(M)
    ##
    warped = kornia.warp_perspective(raw_image, M, dsize=(H,W), align_corners=True)
    mask = kornia.warp_perspective(torch.ones([B,1,H,W], device=device), M, dsize=(H, W), mode='nearest',align_corners=True)
    count = kornia.warp_perspective(torch.ones([B,1,H,W],device=device), M_inv, dsize=(H,W), mode='nearest',align_corners=True)

    # Ignore the detections too close to the border to avoid artifacts
    if config['valid_border_margin']:
        ##TODO: validation & debug
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config['valid_border_margin'] * 2,) * 2)
        kernel = torch.as_tensor(kernel[np.newaxis,:,:], device=device)#BHW
        kernel = torch.flip(kernel, dims=[1,2])
        _, kH, kW = kernel.shape
        origin = ((kH-1)//2, (kW-1)//2)
        count = erosion2d(count, kernel, origin=origin) + 1.
        mask = erosion2d(mask, kernel, origin=origin) + 1.
    mask = mask.squeeze(dim=1)#B,H,W
    count = count.squeeze(dim=1)#B,H,W


    # Predict detection probabilities
    prob = net(warped)


    prob = prob['output']['prob']
    prob = prob * mask
    prob_proj = kornia.warp_perspective(prob.unsqueeze(dim=1), M_inv, dsize=(H,W), align_corners=True)

    prob_proj = prob_proj.squeeze(dim=1)#B,H,W
    prob_proj = prob_proj * count#project back

    #predict_kp
    if(model_name != 'superedgev1'):
        prob_kp = prob['output_kp']['prob']
        prob_kp = prob_kp * mask
        prob_kp_proj = kornia.warp_perspective(prob_kp.unsqueeze(dim=1), M_inv, dsize=(H,W), align_corners=True)
        prob_kp_proj = prob_kp_proj.squeeze(dim=1)#B,H,W
        prob_kp_proj = prob_kp_proj * count#project back
        probs_kp = torch.cat([probs_kp, prob_kp_proj.unsqueeze(dim=1)], dim=1)
    else:
        probs_kp = None
    ##
    # print(probs_kp.shape,prob_kp_proj.shape)
    # print(probs.shape,prob_proj.shape)
    # print("-------------------------")
    #the probabilities of each pixels on raw image 
    probs = torch.cat([probs, prob_proj.unsqueeze(dim=1)], dim=1)#the probabilities of each pixels on raw image
    counts = torch.cat([counts, count.unsqueeze(dim=1)], dim=1)
    images = torch.cat([images, warped.unsqueeze(dim=-1)], dim=-1)

    return probs,probs_kp, counts, images

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
visited_once = []
def find_connected_points(mat,img):
    global paths
    global visited_once
    res = []
    count = 0 
    visited_kp = [[False] * len(mat[0]) for _ in range(len(mat))]
    # 遍历整个矩阵，找到起点
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] in [2, 3]  and visited_kp[i][j] == False:
                # 如果找到了起点，进行深度优先搜索
                visited = [[False] * len(mat[0]) for _ in range(len(mat))]
                dfs(mat, visited,visited_kp, i, j, 0,[])
                res.append(paths)
                paths=[]
    # 如果没有找到起点，返回空列表
    return res
paths = []

def dfs(mat,visited,visited_kp,i,j,depth,path):

    global paths
    if(depth > 150):
        paths.append(path)
        return 
    flag = 0
    visited[i][j] = True
    for x in range(i - 5, i + 6):
        for y in range(j - 5, j + 6):
            if x < 0 or x >= len(mat) or y < 0 or y >= len(mat[0]) or visited[x][y] :
                continue
            if(mat[x][y] in [2,3]):
                visited_kp[x][y] = True
            if(mat[x][y] in [1,2,3]):
                path.append([x,y])
                flag = 1
                dfs(mat,visited,visited_kp,x,y,depth+1,path)
    if(flag == 0):
        if(depth > 5):
            paths.append(path)
    return 

@torch.no_grad()
def homography_adaptation(net, raw_image, config, device='cpu', model_name='superedgev1'):
    """
    :param raw_image: [B,1,H,W]
    :param net: MagicPointNet
    :param config:
    :return:
    """
    probs = net(raw_image)#B,H,W
    #冷凯添加

    if model_name != "superedgev1":
        probs_kp = probs['output_kp']['prob']
    else:
        probs_kp = None
    probs = probs['output']['prob']


    ## probs = torch.tensor(np.load('./prob.npy'), dtype=torch.float32)#debug
    ## warped_prob = torch.tensor(np.load('./warped_prob.npy'), dtype=torch.float32)#debug

    counts = torch.ones_like(probs)
    #TODO: attention dim expand
    probs = probs.unsqueeze(dim=1)
    if probs_kp != None: probs_kp = probs_kp.unsqueeze(dim=1)
    counts = counts.unsqueeze(dim=1)
    images = raw_image.unsqueeze(dim=-1)#maybe no need
    #
    H,W = raw_image.shape[2:4]#H,W
    config = dict_update(homography_adaptation_default_config, config)

    for _ in range(config['num']-1):
        probs, probs_kp ,counts, images = one_adaptation(net, raw_image, probs, probs_kp,counts, images, config, device=device,model_name=model_name)

    counts = torch.sum(counts, dim=1)
    max_prob, _ = torch.max(probs, dim=1)
    if probs_kp != None:
        max_prob_kp, _ = torch.max(probs_kp, dim=1)
    else:
        max_prob_kp = None
    
    mean_prob = torch.sum(probs, dim=1)/counts
    if probs_kp != None:
        mean_prob_kp = torch.sum(probs_kp, dim=1)/counts
    else:
        mean_prob_kp = None
    if config['aggregation'] == 'max':
        print("===========")
        prob = max_prob
        probs_kp = max_prob_kp

    elif config['aggregation'] == 'sum':
        prob = mean_prob
        probs_kp = mean_prob_kp
    else:
        raise ValueError('Unkown aggregation method: {}'.format(config['aggregation']))

    if config['filter_counts']:
        prob = torch.where(counts>=config['filter_counts'], prob, torch.zeros_like(prob))

    return {'prob': prob, 'counts': counts,'prob_kp':probs_kp,'mean_prob_kp':mean_prob_kp,
            'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs}


if __name__=='__main__':
    import matplotlib.pyplot as plt

    # with open('./config/homo_test.yaml', 'r', encoding='utf8') as fin:
    #     config = yaml.safe_load(fin)
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    #配置文件的路径
    args = parser.parse_args()

    config_file = args.config
    assert (os.path.exists(config_file))
    ##
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)


    if not os.path.exists(config['data']['dst_label_path']):
        os.makedirs(config['data']['dst_label_path'])
    if not os.path.exists(config['data']['dst_image_path']):
        os.makedirs(config['data']['dst_image_path'])


    image_list = os.listdir(config['data']['src_image_path'])
    image_list.sort()
    
    image_list = [os.path.join(config['data']['src_image_path'], fname) for fname in image_list]

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # net = MagicPoint(config['model'], input_channel=1, grid_size=8,device=device)
    # net.load_state_dict(torch.load(config['model']['pretrained_model']))
    # net.to(device).eval()
    if config['model']['name'] == 'superedge':
        net = SuperEdge(config['model'], device=device, using_bn=config['model']['using_bn'])
    elif config['model']['name'] == 'superedgev1':
        net = SuperEdgeV1(config['model'], device=device,using_bn=config['model']['using_bn'])
    net.load_state_dict(torch.load(config['model']['pretrained_model']))
    net.to(device).eval()

    batch_fnames,batch_imgs,batch_raw_imgs = [],[],[]

    for idx, fpath in tqdm(enumerate(image_list)):
        root_dir, fname = os.path.split(fpath)
        ##

        img = read_image(fpath)

        img = ratio_preserving_resize(img, config['data']['resize'])
        
        t_img = to_tensor(img, device)
        ##
        batch_imgs.append(t_img)
        batch_fnames.append(fname)
        batch_raw_imgs.append(img)
        ##
        if len(batch_imgs)<1 and ((idx+1)!=len(image_list)):
            continue

        batch_imgs = torch.cat(batch_imgs)
        outputs = homography_adaptation(net, batch_imgs, config['data']['homography_adaptation'], device=device, model_name = config['model']['name'] )
        prob = outputs['prob']
        prob_kp = outputs['prob_kp']
        # if config['model']['nms']:
        #     prob_kp = [box_nms(p.unsqueeze(dim=0),#to 1HW
        #                     config['model']['nms'],
        #                     # 0,
        #                     min_prob=config['model']['det_thresh'],
        #                     keep_top_k=config['model']['topk']).squeeze(dim=0) for p in prob_kp]
        #     prob_kp = torch.stack(prob_kp)
       
        pred = (prob>=config['model']['det_thresh']).int()
        
        # pred_kp = (prob_kp>=config['model']['det_thresh']).int()
        ##
        points = [torch.stack(torch.where(e)).T for e in pred]
        points = [pt.cpu().numpy() for pt in points]
        
        # points_kp = [torch.stack(torch.where(e)).T for e in pred_kp]
        # points_kp = [pt.cpu().numpy() for pt in points_kp]
        
        #------------------------------------------------------------------------------------------------------------------------------------
        ##save points
        for fname, pt in zip(batch_fnames, points):
            cv2.imwrite(os.path.join(config['data']['dst_image_path'], fname), img)
            np.save(os.path.join(config['data']['dst_label_path'], fname+'.npy'), pt)
            # np.save(os.path.join(config['data']['dst_label_path'],"kp"+ fname+'.npy'), kp_pt)
            print('{}, {}'.format(os.path.join(config['data']['dst_label_path'], fname+'.npy'), len(pt)))

        # ## debug
        # for img, pts in zip(batch_raw_imgs,points):
        #     debug_img = cv2.merge([img, img, img])
        #     for pt in pts:
        #         cv2.circle(debug_img, (int(pt[1]),int(pt[0])), 1, (0,255,0), thickness=-1)
        #     plt.imshow(debug_img)
        #     plt.show()
        # if idx>=2:
        #     break
        batch_fnames,batch_imgs,batch_raw_imgs = [],[],[]
    print('Done')
