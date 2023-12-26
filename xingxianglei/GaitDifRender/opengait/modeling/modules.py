import torch
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer
from utils import clones, is_list_or_tuple


class HorizontalPoolingPyramid(nn.Cell):
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        super().__init__()
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def construct(self, x: ms.Tensor):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.shape[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return ops.cat(features, -1)


class SetBlockWrapper(nn.Cell):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def construct(self, x: ms.Tensor, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.shape
        x = self.forward_block(x.swapaxes(
            1, 2).reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.shape
        return x.reshape(n, s, *output_size[1:]).swapaxes(1, 2)


class PackSequenceWrapper(nn.Cell):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def construct(self, seqs: ms.Tensor, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, pad_mode="pad", padding=padding, has_bias=False, **kwargs)

    def construct(self, x: ms.Tensor):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Cell):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = ms.Parameter(default_input=initializer('XavierUniform', [parts_num, in_channels, out_channels]))
        self.l2_normalize = ops.L2Normalize(axis=1)
        self.norm = norm

    def construct(self, x: ms.Tensor):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1)
        if self.norm:
            out = x.matmul(self.l2_normalize(self.fc_bin))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0)


class SeparateBNNecks(nn.Cell):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm

        self.fc_bin = ms.Parameter(default_input=initializer('XavierUniform', [parts_num, in_channels, class_num]))

        # TODO：SeparateBNNecks 并行运算
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d
        self.l2_normalize = ops.L2Normalize(axis=1)
        self.feature_l2_normalize = ops.L2Normalize(axis=-1)

    def construct(self, x: ms.Tensor):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.shape
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = ms.ops.cat([bn(_x) for _x, bn in zip(x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1)
        if self.norm:
            feature = self.feature_l2_normalize(feature)  # [p, n, c]

            logits = feature.matmul(self.l2_normalize(self.fc_bin))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0), logits.permute(1, 2, 0)


class FocalConv2d(nn.Cell):
    """
        GaitPart: Temporal Part-based Model for Gait Recognition
        CVPR2020: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
        Github: https://github.com/ChaoFan96/GaitPart
    """

    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, has_bias=False, **kwargs)

    def construct(self, x: ms.Tensor):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.shape[2]
            split_size = int(h // 2 ** self.halving)
            z = x.split(split_size, 2)
            z = ms.ops.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), pad_mode='pad',
                 padding=(1, 1, 1, 1, 1, 1),
                 bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, pad_mode=pad_mode, padding=padding, has_bias=bias, **kwargs)

    def construct(self, ipts: ms.Tensor):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


# TODO:重载GaitAlign
#
# class GaitAlign(nn.Cell):
#     """
#         GaitEdge: Beyond Plain End-to-end Gait Recognition for Better Practicality
#         ECCV2022: https://arxiv.org/pdf/2203.03972v2.pdf
#         Github: https://github.com/ShiqiYu/OpenGait/tree/master/configs/gaitedge
#     """
#
#     def __init__(self, H=64, W=44, eps=1, **kwargs):
#         super(GaitAlign, self).__init__()
#         self.H, self.W, self.eps = H, W, eps
#         self.Pad = nn.ZeroPad2d((int(self.W / 2), int(self.W / 2), 0, 0))
#         # self.RoiPool = RoIAlign((self.H, self.W), 1, sampling_ratio=-1)
#         self.RoiPool = ops.ROIAlign(pooled_height=self.H,
#                                     pooled_width=self.W,
#                                     spatial_scale=1,
#                                     sample_num=-1,
#                                     roi_end_mode=0)
#
#     def __call__(self, feature_map, binary_mask, w_h_ratio):
#         """
#            In  sils:         [n, c, h, w]
#                w_h_ratio:    [n, 1]
#            Out aligned_sils: [n, c, H, W]
#         """
#         n, c, h, w = feature_map.shape
#         # w_h_ratio = w_h_ratio.repeat(1, 1) # [n, 1]
#         w_h_ratio = w_h_ratio.view(-1, 1)  # [n, 1]
#
#         h_sum = binary_mask.sum(-1)  # [n, c, h]
#         _ = (h_sum >= self.eps).float().cumsum(axis=-1)  # [n, c, h]
#         h_top = (_ == 0).float().sum(-1)  # [n, c]
#         h_bot = (_ != torch.max(_, dim=-1, keepdim=True)
#         [0]).float().sum(-1) + 1.  # [n, c]
#
#         w_sum = binary_mask.sum(-2)  # [n, c, w]
#         w_cumsum = w_sum.cumsum(axis=-1)  # [n, c, w]
#         w_h_sum = w_sum.sum(-1).unsqueeze(-1)  # [n, c, 1]
#         w_center = (w_cumsum < w_h_sum / 2.).float().sum(-1)  # [n, c]
#
#         p1 = self.W - self.H * w_h_ratio
#         p1 = p1 / 2.
#         p1 = torch.clamp(p1, min=0)  # [n, c]
#         t_w = w_h_ratio * self.H / w
#         p2 = p1 / t_w  # [n, c]
#
#         height = h_bot - h_top  # [n, c]
#         width = height * w / h  # [n, c]
#         width_p = int(self.W / 2)
#
#         feature_map = self.Pad(feature_map)
#         w_center = w_center + width_p  # [n, c]
#
#         w_left = w_center - width / 2 - p2  # [n, c]
#         w_right = w_center + width / 2 + p2  # [n, c]
#
#         w_left = torch.clamp(w_left, min=0., max=w + 2 * width_p)
#         w_right = torch.clamp(w_right, min=0., max=w + 2 * width_p)
#
#         boxes = torch.cat([w_left, h_top, w_right, h_bot], dim=-1)
#         # index of bbox in batch
#         box_index = torch.arange(n, device=feature_map.device)
#         rois = torch.cat([box_index.view(-1, 1), boxes], -1)
#         crops = self.RoiPool(feature_map, rois)  # [n, c, H, W]
#         return crops


# def RmBN2dAffine(model):
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.weight.requires_grad = False
#             m.bias.requires_grad = False
