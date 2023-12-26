from numpy.core.fromnumeric import size
import mindspore
from mindspore import nn, ops
import mindspore.ops.operations as P
from mindspore import Tensor
import numpy as np
from mindspore import dtype as mstype
class ConvBNReLU(nn.Cell):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                    has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                    c_in, c_out, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, has_bias=False, group=group,pad_mode='pad')
            norm_layer = nn.BatchNorm2d
        elif mode == '1d':
            self.conv = nn.Conv1d(
                    c_in, c_out, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, has_bias=False, group=group,pad_mode='pad')
            norm_layer = nn.BatchNorm1d
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

class Stat_fea(nn.Cell):
    def __init__(self, level_num, input_dim):
        super(Stat_fea, self).__init__()
        self.conv1 = nn.SequentialCell(ConvBNReLU(input_dim, 256, 3, 1, 1, has_relu=False), nn.LeakyReLU())
        self.conv2 = ConvBNReLU(256, 128, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.SequentialCell(ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'), nn.LeakyReLU())
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        self.out = ConvBNReLU(256, 128, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num
    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        N, C, H, W = x.shape
        x_ave =ops.adaptive_avg_pool2d(x, (1, 1))
        l2_normalize = ops.L2Normalize(axis=1)

        cos_sim = (l2_normalize(x_ave) * l2_normalize(x)).sum(1)
        cos_sim = cos_sim.view(N, -1)

        reduce_min_op = P.ReduceMin()
        cos_sim_min = reduce_min_op(cos_sim, -1)
        # cos_sim_min, _ = cos_sim.min(-1)
        cos_sim_min = cos_sim_min.unsqueeze(-1)

        reduce_max_op = P.ReduceMax()
        cos_sim_max = reduce_max_op(cos_sim,-1)
        # cos_sim_max, _ = cos_sim.max(-1)
        cos_sim_max = cos_sim_max.unsqueeze(-1)
        q_levels = ops.arange(self.level_num).float()

        shape = (N, self.level_num)
        q_levels = q_levels.broadcast_to(shape)
        q_levels =  (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min
        q_levels_feas = q_levels
        q_levels = q_levels.unsqueeze(1)
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
        q_levels_inter = q_levels_inter.unsqueeze(-1)
        cos_sim = cos_sim.unsqueeze(-1)
        quant = 1 - ops.abs(q_levels - cos_sim)
        quant = quant * (quant > (1 - q_levels_inter))
        sta = quant.sum(1)
        sta = sta / (sta.sum(-1).unsqueeze(-1))
        sta_feas = sta
        return sta_feas, q_levels_feas