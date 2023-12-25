import torch
from model.backbone import resnet
import numpy as np
import sys
import os
# sys.path.insert(1, os.path.join(sys.path[0], '/content/efficientvit'))
sys.path.insert(1, os.path.join(sys.path[0], '/kaggle/working/efficientvit'))

from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0
from efficientvit.models.efficientvit.cls import ClsHead
from efficientvit.models.efficientvit.seg import SegHead

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(37, 10, 4), use_aux=False):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        # input : nchw,
        # output: (w+1) * sample_rows * 4 
        #self.model = resnet(backbone, pretrained=pretrained)

        # self.model =  efficientvit_backbone_b0()

        self.model = efficientvit_backbone_b0()

        if self.use_aux:
            self.aux_header = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=32,
            head_depth=2,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=cls_dim[-1] + 1,
            # **build_kwargs_from_config(kwargs, SegHead),
        )
            
            initialize_weights(self.aux_header)

        # self.cls = torch.nn.Sequential(
        #     torch.nn.Linear(1800, 2048),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(2048, self.total_dim),
        # )


        self.cls=head = ClsHead(
        in_channels=128,
        width_list=[1024, 1280],
        n_classes=self.total_dim,
        # **build_kwargs_from_config(kwargs, ClsHead),
        ) 

        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18'] else torch.nn.Conv2d(128,8,1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4
        initialize_weights(self.cls)

    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        # x2,x3,fea = self.model(x)

        outs = self.model(x)
        # fea=outs['stage4']
        # x3=outs['stage3']
        # x2=outs['stage2']
        if self.use_aux:
            aux_seg  = self.aux_header(outs)
        else:
            aux_seg = None

        #fea = self.pool(fea).view(-1, 1800)

        group_cls = self.cls(outs).view(-1, *self.cls_dim)
        # print(group_cls.shape)

        if self.use_aux:
            return group_cls, aux_seg

        return group_cls


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)
