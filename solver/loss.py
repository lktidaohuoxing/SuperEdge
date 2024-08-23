#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn.functional as F
from utils.keypoint_op import warp_points
from utils.tensor_op import pixel_shuffle_inv
from utils.tensor_op import pixel_shuffle

def loss_func(config,model, data, prob, prob_warp=None, device='cuda:1'):
    if model['name'] == 'superedge':
        prob_kp = prob['output_kp']
    prob = prob['output']
   
    #edge lossprob = prob
    # print(prob['logits'].shape)


    pix_edge_loss = detector_loss(data['raw']['kpts_map'],
                            prob['logits'],
                            data['raw']['mask'],
                            config['grid_size'],
                            device=device)
    if model['name'] == 'superedgev1':
        return pix_edge_loss
    obj_edge_loss = detector_loss_weight(data['raw']['kp_kpts_map'],
                            prob_kp['logits'],
                            data['raw']['mask'],
                            config['grid_size'],
                            device=device)

    return pix_edge_loss+ obj_edge_loss
   

def bdcn_lossORI(inputs, targets ,cuda=False,device = 'cuda:0'):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    # print(cuda)
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * 1.1 / valid  # balance = 1.1
    weights = torch.Tensor(weights)
    # if cuda:
    weights = weights.cuda(device=device)
    inputs = torch.sigmoid(inputs)
    loss = torch.nn.BCELoss(weights, reduction='sum')(inputs.float(), targets.float())
    return loss

def bdcn_loss2(inputs, targets,valid_mask=None, grid_size=8, l_weight=64, device = 'cuda:0'):
    
    valid_mask = torch.ones_like(targets) if valid_mask is None else valid_mask
    valid_mask = valid_mask.unsqueeze(dim=1)

    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()

    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.0 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    inputs= torch.sigmoid(inputs)
    loss = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())

    loss = torch.divide(torch.sum(loss*valid_mask , dim=(1, 2)), torch.sum(valid_mask + 1e-6, dim=(1, 2)))
    loss = torch.mean(loss)

    return l_weight*loss

def detector_loss(keypoint_map, logits, valid_mask=None, grid_size=8, device='cpu'):
    """
    :param keypoint_map: [B,H,W]
    :param logits: [B,65,Hc,Wc]
    :param valid_mask:[B, H, W]
    :param grid_size: 8 default
    :return:
    """
    # print(keypoint_map.shape, logits.shape)
    # Convert the boolean labels to indices including the "no interest point" dustbin
    labels = keypoint_map.unsqueeze(1).float()#to [B, 1, H, W]
    labels = pixel_shuffle_inv(labels, grid_size) # to [B,64,H/8,W/8]
    B,C,h,w = labels.shape#h=H/grid_size,w=W/grid_size
    labels = torch.cat([2*labels, torch.ones([B,1,h,w],device=device)], dim=1)
    # Add a small random matrix to randomly break ties in argmax
    labels = torch.argmax(labels + torch.zeros(labels.shape,device=device).uniform_(0,0.1),dim=1)#B*65*Hc*Wc

    # Mask the pixels if bordering artifacts appear
    valid_mask = torch.ones_like(keypoint_map) if valid_mask is None else valid_mask

    valid_mask = valid_mask.unsqueeze(1)
    valid_mask = pixel_shuffle_inv(valid_mask, grid_size)#[B, 64, H/8, W/8]
    valid_mask = torch.prod(valid_mask, dim=1).unsqueeze(dim=1).type(torch.float32)#[B,1,H/8,W/8]

    ## method 1
    ce_loss = F.cross_entropy(logits, labels, reduction='none',)
    # print(ce_loss.shape)
    valid_mask = valid_mask.squeeze(dim=1)
    loss = torch.divide(torch.sum(ce_loss * valid_mask, dim=(1, 2)), torch.sum(valid_mask + 1e-6, dim=(1, 2)))
    loss = torch.mean(loss)

    ## method 2
    ## method 2 equals to tf.nn.sparse_softmax_cross_entropy()
    # epsilon = 1e-6
    # loss = F.log_softmax(logits,dim=1)
    # mask = valid_mask.type(torch.float32)
    # mask /= (torch.mean(mask)+epsilon)
    # loss = torch.mul(loss, mask)
    # loss = F.nll_loss(loss,labels)
    return loss
    
def detector_loss_weight(keypoint_map, logits, valid_mask=None, grid_size=8, device='cpu'):
    """
    :param keypoint_map: [B,H,W]
    :param logits: [B,65,Hc,Wc]
    :param valid_mask:[B, H, W]
    :param grid_size: 8 default
    :return:
    """
    # print(keypoint_map.shape, logits.shape)
    # Convert the boolean labels to indices including the "no interest point" dustbin

    labels = keypoint_map.unsqueeze(1).float()#to [B, 1, H, W]
    labels = pixel_shuffle_inv(labels, grid_size) # to [B,64,H/8,W/8]

    B,C,h,w = labels.shape#h=H/grid_size,w=W/grid_size
    labels = torch.cat([2*labels, torch.ones([B,1,h,w],device=device)], dim=1)

    # Add a small random matrix to randomly break ties in argmax
    labels = torch.argmax(labels + torch.zeros(labels.shape,device=device).uniform_(0,0.1),dim=1)#B*65*Hc*Wc
    
    # Mask the pixels if bordering artifacts appear
    valid_mask = torch.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = valid_mask.unsqueeze(1)
    valid_mask = pixel_shuffle_inv(valid_mask, grid_size)#[B, 64, H/8, W/8]
    valid_mask = torch.prod(valid_mask, dim=1).unsqueeze(dim=1).type(torch.float32)#[B,1,H/8,W/8]

    

    # print(mask.shape,valid_mask.squeeze(dim=1).shape)
    # mask = mask * (valid_mask.squeeze(dim=1)+ 1e-7)
    mask = labels.float()
    num_positive = torch.sum((mask < 64).float()).float() # >0.1
    num_negative = torch.sum((mask == 64).float()).float() # <= 0.1
    mask[mask < 64] = (1.0 * num_negative / (num_positive + num_negative)).float() #0.1
    mask[mask == 64] = (1.0 * num_positive / (num_positive + num_negative)).float() # before mask[mask <= 0.1]

    # mask[mask == 2] = 0
    # inputs= torch.sigmoid(inputs)
    # loss = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    ## method 1
    ce_loss = F.cross_entropy(logits, labels, reduction='none',)
    #add weight 
    ce_loss = ce_loss * mask

    # print(ce_loss.shape)
    valid_mask = valid_mask.squeeze(dim=1)
    loss = torch.divide(torch.sum(ce_loss * valid_mask, dim=(1, 2)), torch.sum(valid_mask + 1e-6, dim=(1, 2)))
    loss = torch.mean(loss)

    return loss


def inline_descriptor_loss(config,descriptors,warped_descriptores,kp_map,device='cuda:1'):
    # print("kp_map shape : ",kp_map.shape)
    # print("discriptors shape : ",descriptors.shape)
    # print("warped_descriptores shape : ",warped_descriptores.shape)
    (batch_size, D, Hc, Wc) = descriptors.shape
    mask = kp_map > 0.015
    # mask = mask.unsqueeze(dim=1)
    #正样本
    descriptors = torch.reshape(descriptors, [batch_size, Hc, Wc,D])
    descriptors = F.normalize(descriptors, p=2, dim=2)

    dist = torch.cdist(descriptors[mask], descriptors[mask], p=2)

    # dist_matrix = torch.zeros_like(kp_map, dtype=torch.float32)
    # dist_matrix[mask] = dist.view(mask.sum(), -1).mean(dim=1)

    #---------------------------------------
    inv_mask = ~mask
    descriptors_inv = descriptors[inv_mask]
    dist_inv = torch.cdist(descriptors_inv, descriptors[mask], p=2)
    # print( torch.mean(dist) )
    # print(torch.mean(dist_inv))


    loss =  torch.mean(dist_inv)/ torch.mean(dist) 
    # print(loss)
    return loss

