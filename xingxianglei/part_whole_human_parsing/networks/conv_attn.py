from mindspore import ops, Parameter
import mindspore.nn as nn
from networks.weight_scaled_networks import WeightScaledConv2d


class ConvAttn(nn.Cell):
    def __init__(self, in_channels, num_parts=14, feat_dim=64):
        super().__init__()
        self.num_parts = num_parts
        self.feat_dim = feat_dim
        self.attn = WeightScaledConv2d(in_channels, num_parts, 1)
        self.feat = WeightScaledConv2d(in_channels, num_parts * feat_dim, 1)
        self.bg_weight = Parameter(ops.zeros([1]))

    def construct(self, x):
        batch_size, _, h, w = x.shape
        attn = self.attn(x)
        bg_attn = self.bg_weight * ops.ones([batch_size, 1, h, w])
        all_attn = ops.cat([attn, bg_attn], axis=1)
        all_attn = ops.softmax(all_attn, axis=1)
        attn = all_attn[:, :-1]
        feat = self.feat(x)
        feat = feat.reshape([batch_size, self.num_parts, self.feat_dim, h, w])
        feat = feat * attn[:, :, None]
        # feat = torch.mean(feat, dim=(3, 4))
        return feat, attn
