# -*-coding:utf8-*-
import torch
from utils.tensor_op import pixel_shuffle,pixel_shuffle_inv
import torch.nn.functional as F
from torch import nn

class PixelHead(torch.nn.Module):
    def __init__(self, input_channel, grid_size, using_bn=True):
        super(PixelHead, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn
        ##
        self.convPa = torch.nn.Conv2d(input_channel, 256, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convPb = torch.nn.Conv2d(256, pow(grid_size, 2)+1, kernel_size=1, stride=1, padding=0)

        self.bnPa,self.bnPb = None,None
        if using_bn:
            self.bnPa = torch.nn.BatchNorm2d(256)
            self.bnPb = torch.nn.BatchNorm2d(pow(grid_size, 2)+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnPa(self.relu(self.convPa(x)))
            # print("decoder1",out.shape)
            out = self.bnPb(self.convPb(out))  #(B,65,H,W)
            # print("decoder2",out.shape)
        else:
            out = self.relu(self.convPa(x))
            out = self.convPb(out)  # (B,65,H,W)
        
        prob = self.softmax(out)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        # Reshape to get full resolution heatmap.
        prob = pixel_shuffle(prob, self.grid_size)  # [B,1,H*8,W*8]
        prob = prob.squeeze(dim=1)#[B,H,W]
        return {'logits':out, 'prob':prob}

class ObjectHead(torch.nn.Module):
    def __init__(self, input_channel, grid_size, using_bn=True):
        super(ObjectHead, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn
        ##
        self.convPa = torch.nn.Conv2d(input_channel, 256, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        #QKV head
        self.convQ = nn.Sequential(nn.Conv2d(256, (pow(grid_size, 2)+1),kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d((pow(grid_size, 2)+1)) )
        self.convK = nn.Sequential(nn.Conv2d(256, (pow(grid_size, 2)+1),kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d((pow(grid_size, 2)+1)) )
        self.convV = nn.Sequential(nn.Conv2d(256, (pow(grid_size, 2)+1),kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d((pow(grid_size, 2)+1)) )


        self.convPb = torch.nn.Conv2d(pow(grid_size, 2)+1, pow(grid_size, 2)+1, kernel_size=1, stride=1, padding=0)

        self.bnPa,self.bnPb = None,None
        if using_bn:
            self.bnPa = torch.nn.BatchNorm2d(256)
            self.bnPb = torch.nn.BatchNorm2d(pow(grid_size, 2)+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnPa(self.relu(self.convPa(x)))
            out_q = self.convQ(out)
            batch_size, channels, height, width = out_q.shape
            out_k = self.convK(out)
            out_v = self.convV(out)
            out_q = out_q.view(batch_size, channels, -1)
            out_k = out_k.view(batch_size, channels, -1)
            out_v = out_v.view(batch_size, channels, -1)
            out_k = out_k.transpose(1, 2)  # Now out_k is [batch_size, height*width, channels]
            # Calculate the attention weights
            weights = F.softmax(torch.bmm(out_q, out_k), dim=-1)
            # Apply the attention weights to V
            out_v = torch.bmm(weights, out_v)  # out_v shape [batch_size, channels, height*width]
            # Reshape out_v to have the same shape as the input
            out_v = out_v.view(batch_size, channels, height, width)
            out = self.bnPb(self.convPb(out_v))  #(B,65,H,W)

        else:
            out = self.relu(self.convPa(x))
            out = self.convPb(out)  # (B,65,H,W)

        prob = self.softmax(out)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        # Reshape to get full resolution heatmap.
        prob = pixel_shuffle(prob, self.grid_size)  # [B,1,H*8,W*8]
        prob = prob.squeeze(dim=1)#[B,H,W]
        return {'logits':out, 'prob':prob}
