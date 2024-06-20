import torch
import torch.nn as nn
import torch.nn.init as init
from spconv.pytorch import SparseConvTensor

import torch.nn.functional as F


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 256]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 1024, 3, 1)
        elif level==1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level==2:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0) # [B, 512, 100, 88] -> [B, 256, 100, 88]
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest') # [B, 256, 100, 88] -> [B, 256, 200, 176]
            level_1_resized =x_level_1 # [B, 256, 200, 176]
            level_2_resized =self.stride_level_2(x_level_2) # [B, 256, 400, 352] -> [B, 256, 200, 176]
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized) # [B, 16, 200, 176]
        level_1_weight_v = self.weight_level_1(level_1_resized) # [B, 16, 200, 176]
        level_2_weight_v = self.weight_level_2(level_2_resized) # [B, 16, 200, 176]
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1) # [B, 48, 200, 176]
        levels_weight = self.weight_levels(levels_weight_v) # [B, 3, 200, 176]
        levels_weight = F.softmax(levels_weight, dim=1) # [B, 3, 200, 176]

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:] # [B, 256, 200, 176]

        out = self.expand(fused_out_reduced) # [B, 256, 200, 176] -> [B, 512, 200, 176]

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class SMFFNeck(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
    
        num_filters = model_cfg.num_filters
        layer_nums = model_cfg.layer_nums
        in_channels = model_cfg.in_channels

        self.uphead = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, 2, stride=2, bias=False),
            nn.BatchNorm2d(num_filters, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.block_5 = self._build_layer(
            in_channels[0], in_channels[0], layer_nums[0], stride=1)
        
        self.block_4 = self._build_layer(
            in_channels[1] , in_channels[1], layer_nums[1], stride=1)
        
        self.block_3 = self._build_layer(
            in_channels[2] , in_channels[2], layer_nums[2], stride=1)
        
        self.asff = ASFF(level=1)

        self.num_bev_features = num_filters

    def downsample_factor(self):
        return 1

    def _build_layer(self, inplanes, planes, num_blocks, stride=1):

        block = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU())

        for j in range(num_blocks):
            block.add_module(f'spconv_{num_blocks}', nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add_module(f'BN_{num_blocks}', nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),)
            block.add_module(f'RL_{num_blocks}', nn.ReLU())

        return block
 
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m)

    def forward(self, data_dict):
        x_conv3 = data_dict['dense_feature']['densev3'] # [B, 400, 352]
        x_conv4 = data_dict['dense_feature']['densev4'] # [B, 200, 176]
        x_conv5 = data_dict['dense_feature']['densev5'] # [B, 100, 88]
        if isinstance(x_conv3, SparseConvTensor):
            x_conv3 = x_conv3.dense()
            N, C, D, H, W = x_conv3.shape
            x_conv3 = x_conv3.view(N, C * D, H, W) # [B, 256, 400, 352]
        if isinstance(x_conv4, SparseConvTensor):
            x_conv4 = x_conv4.dense()
            N, C, D, H, W = x_conv4.shape
            x_conv4 = x_conv4.view(N, C * D, H, W) # [B, 256, 200, 176]
        if isinstance(x_conv5, SparseConvTensor):
            x_conv5 = x_conv5.dense()
            N, C, D, H, W = x_conv5.shape
            x_conv5 = x_conv5.view(N, C * D, H, W) # [B, 512, 100, 88]

        x_conv5 = self.block_5(x_conv5) # [B, 512, 100, 88]
        x_conv4 = self.block_4(x_conv4) # [B, 256, 200, 176]
        x_conv3 = self.block_3(x_conv3) # [B, 256, 400, 352]

        x = self.asff(x_conv5, x_conv4, x_conv3) # [B, 512, 200, 176]

        up_x = self.uphead(x) # [B, 512, 400, 352]

        data_dict['spatial_features_2d'] = up_x
        
        return data_dict