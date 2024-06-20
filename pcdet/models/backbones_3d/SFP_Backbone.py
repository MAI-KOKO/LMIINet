import torch.nn as nn
from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
from .spconv_backbone import post_act_block, SparseBasicBlock


class SEDBlock(spconv.SparseModule):

    def __init__(self, dim, kernel_size, stride, num_SBB, norm_fn, indice_key):
        super(SEDBlock, self).__init__()

        first_block = post_act_block( # 论文中的Down模块
            dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            norm_fn=norm_fn, indice_key=f'spconv_{indice_key}', conv_type='spconv')

        block_list = [first_block if stride > 1 else nn.Identity()] # nn.Identity 将输入原样返回
        for _ in range(num_SBB): # num_SparseBasicBlock 即论文中的m x SSR
            block_list.append(
                SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key=indice_key)) # 两个子流形卷积

        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, x):
        return self.blocks(x)


class SEDLayer(spconv.SparseModule): # 不改变通道数和尺寸

    def __init__(self, dim: int, down_kernel_size: list, down_stride: list, num_SBB: list, norm_fn, indice_key):
        super().__init__()

        assert down_stride[0] == 1 # hard code
        assert len(down_kernel_size) == len(down_stride) == len(num_SBB)

        self.encoder = nn.ModuleList()
        for idx in range(len(down_stride)):
            self.encoder.append(
                SEDBlock(dim, down_kernel_size[idx], down_stride[idx], num_SBB[idx], norm_fn, f"{indice_key}_{idx}"))

        downsample_times = len(down_stride[1:]) # 第一层尺度不变，下采样倍率从第二层开始
        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.ups = nn.ModuleList()
        for idx, kernel_size in enumerate(down_kernel_size[1:]): # encoder有三层，第一层不改变尺寸，decoder有两层，尺寸都改变
            self.decoder.append( # 逆卷积只有在与卷积同一个indice_key下才能进行，并且kernel_size要和下采样时一致，没有stride、padding参数
                # 上采样的 f"spconv_{indice_key}_{downsample_times - idx} 与 SEDBlock下采样的 f'spconv_{indice_key} 最终形式一样 表达了在第几个sedlayer层的第几个idx采样层
                post_act_block(
                    dim, dim, kernel_size, norm_fn=norm_fn, conv_type='inverseconv',
                    indice_key=f"spconv_{indice_key}_{downsample_times - idx}"))
            self.ups.append(SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key=f"ups_{indice_key}_{downsample_times - idx}"))
            self.decoder_norm.append(norm_fn(dim))

    def forward(self, x):
        features = []
        for conv in self.encoder:
            x = conv(x)
            features.append(x)

        x = features[-1]
        for deconv, upconv, norm, up_x in zip(self.decoder, self.ups, self.decoder_norm, features[:-1][::-1]):
            x = deconv(x)
            x_res = upconv(up_x)
            x = replace_feature(x, x.features + x_res.features)
            x = replace_feature(x, norm(x.features))
        return x


class SFPBackbone(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        dim = model_cfg.FEATURE_DIM
        num_layers = model_cfg.NUM_LAYERS
        num_SBB = model_cfg.NUM_SBB
        down_kernel_size = model_cfg.DOWN_KERNEL_SIZE
        down_stride = model_cfg.DOWN_STRIDE

        # [41, 1600, 1408] -> [21, 800, 704]
        self.conv1 = spconv.SparseSequential(
            post_act_block(input_channels, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='subm'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='stem'), # 由两个子流形卷积构成
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='stem'),
            post_act_block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
        )

        # [21, 800, 704] -> [11, 400, 352]
        self.conv2 = spconv.SparseSequential(
            SEDLayer(32, down_kernel_size, down_stride, num_SBB, norm_fn=norm_fn, indice_key='sedlayer2_1'),
            SEDLayer(32, down_kernel_size, down_stride, num_SBB, norm_fn=norm_fn, indice_key='sedlayer2_2'),
            post_act_block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        )

        #  [11, 400, 352] -> [11, 200, 176]
        self.conv3 = spconv.SparseSequential(
            SEDLayer(64, down_kernel_size, down_stride, num_SBB, norm_fn=norm_fn, indice_key='sedlayer3_1'),
            SEDLayer(64, down_kernel_size, down_stride, num_SBB, norm_fn=norm_fn, indice_key='sedlayer3_2'),
            post_act_block(64, dim, 3, norm_fn=norm_fn, stride=(1, 2, 2), padding=1, indice_key='spconv3', conv_type='spconv'),
        )

        # [11, 200, 176] -> [5, 200, 176]
        self.conv4 = spconv.SparseSequential(
            spconv.SparseConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key='spconv4'),
            norm_fn(dim),
            nn.ReLU(),
        )

        # [11, 400, 352] -> [2, 400, 352]
        self.densev3 = spconv.SparseSequential(
            post_act_block(dim, dim, (3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=0, indice_key='densev3_1', conv_type='spconv'),
            spconv.SparseConv3d(dim, dim*2, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key='densev3_2'),
            norm_fn(dim*2),
            nn.ReLU(),
        )

        # [11, 200, 176] -> [2, 200, 176]
        self.densev4 = spconv.SparseSequential(
            post_act_block(dim, dim, (3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=0, indice_key='densev4_1', conv_type='spconv'),
            spconv.SparseConv3d(dim, dim*2, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key='densev4'),
            norm_fn(dim*2),
            nn.ReLU(),
        )

        # [5, 200, 176] -> [2, 100, 88]
        self.densev5 = spconv.SparseSequential(
            post_act_block(dim, dim*2, (3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=0, indice_key='densev5_1', conv_type='spconv'),
            spconv.SparseConv3d(dim*2, dim*4, 3, stride=(1, 2, 2), padding=1, bias=False, indice_key='densev5_2'),
            norm_fn(dim*4),
            nn.ReLU(),
        )

        self.num_point_features = dim

        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(x) # [21, 800, 704]
        x_conv2 = self.conv2(x_conv1) # [11, 400, 352]
        x_conv3 = self.conv3(x_conv2) # [11, 200, 176]
        x_conv4 = self.conv4(x_conv3) # [5, 200, 176]

        densev3 = self.densev3(x_conv2) # [2, 400, 352] feature_dim: 128
        densev4 = self.densev4(x_conv3) # [2, 200, 176] feature_dim: 128
        densev5 = self.densev5(x_conv4) # [2, 100, 88] fefature_dim：256

        batch_dict.update({
            'dense_feature': {
                'densev3': densev3,
                'densev4': densev4,
                'densev5': densev5
            }
        })

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 2,
                'x_conv2': 4,
                'x_conv3': 8,
                'x_conv4': 8,
            }
        })
        return batch_dict





