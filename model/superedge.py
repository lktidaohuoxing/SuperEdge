#-*-coding:utf8-*-
import torch
from solver.nms import box_nms
from model.modules.cnn.vgg_backbone import VGGBackboneBN,VGGBackbone
from model.modules.cnn.cnn_heads import PixelHead,ObjectHead
import cv2


class SuperEdge(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, config, input_channel=1, grid_size=8, using_bn=True, device='cpu'):
        super(SuperEdge, self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']
        if using_bn:
            self.backbone = VGGBackboneBN(config['backbone']['vgg'], input_channel, device=device)
        else:
            self.backbone = VGGBackbone(config['backbone']['vgg'], input_channel, device=device)

        self.pixel_head = PixelHead(input_channel=128, grid_size=grid_size,using_bn=using_bn)
        self.object_head = ObjectHead(input_channel=128, grid_size=grid_size,using_bn=using_bn)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        """
        if isinstance(x, dict):
            feat_map = self.backbone(x['img'])
        else:
            feat_map = self.backbone(x)
        # print("featmap",feat_map.shape)
        outputs = self.pixel_head(feat_map)
        outputs_kp = self.object_head(feat_map)
        
        prob = outputs['prob']
        prob_kp = outputs_kp['prob']
        # print(prob.shape)
        pred = prob[prob>=self.det_thresh]
        pred_kp = prob_kp[prob_kp>self.det_thresh]
        
        #牢记， prob对应的是像素级边缘， prob_kp 对应的是物体级别边缘
        outputs.setdefault('pred', pred)
        outputs_kp.setdefault('pred_kp', pred_kp)

        return {'output':outputs,'output_kp':outputs_kp}

