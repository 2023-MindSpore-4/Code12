from mindspore import ops, Parameter, Tensor
import mindspore.nn as nn
from networks.weight_scaled_networks import WeightScaledConv2d


def exists(val):
    return val is not None


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)


class Blur(nn.Cell):
    def __init__(self):
        super().__init__()
        weight = Parameter(Tensor([
            [1., 2., 1.],
            [2., 4., 2.],
            [1., 2., 1.]
        ]).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.weight = weight / ops.sum(weight)

    def construct(self, x):
        b, c, h, w = x.shape
        weight = self.weight.repeat(c, 0).repeat(c, 1)
        return ops.conv2d(x, weight, pad_mode='pad', padding=1)


class RGBBlock(nn.Cell):
    def __init__(self, input_channel, upsample, rgba):
        super().__init__()

        out_filters = 3 if not rgba else 4
        self.conv = WeightScaledConv2d(input_channel, out_filters, 1, 1)
        # self.conv = nn.SequentialCell(
        #     WeightScaledConv2d(input_channel, out_filters, 1, 1),
        #     nn.Tanh()
        # )
        # self.conv = nn.SequentialCell(
        #     nn.Conv2d(input_channel, out_filters, 1),
        #     nn.Tanh()
        # )

        self.upsample = upsample
        self.blur = Blur() if upsample else None

    def construct(self, x, prev_rgb):
        b, c, h, w = x.shape
        x = self.conv(x)

        if exists(prev_rgb):
            x = x + prev_rgb

        if self.upsample:
            x = ops.interpolate(x, size=(2 * h, 2 * w), mode='bilinear', align_corners=False)
            x = self.blur(x)
        else:
            x = ops.tanh(x)
        return x


class GeneratorBlock(nn.Cell):
    def __init__(self, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.blur = Blur()

        self.conv1 = nn.SequentialCell(
            WeightScaledConv2d(input_channels, filters, 1, 1, bias=False),
            nn.BatchNorm2d(filters),
        )
        self.conv2 = nn.SequentialCell(
            WeightScaledConv2d(filters, filters, 1, 1, bias=False),
            nn.BatchNorm2d(filters),
        )

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(filters, upsample_rgb, rgba)

    def construct(self, x, prev_rgb=None):
        b, c, h, w = x.shape
        if self.upsample:
            x = ops.interpolate(x, size=(2 * h, 2 * w), mode='bilinear', align_corners=False)
            x = self.blur(x)

        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        rgb = self.to_rgb(x, prev_rgb)
        return x, rgb


class ResDownBlock(nn.Cell):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.SequentialCell(
            WeightScaledConv2d(input_channels, filters, 1, stride=2, bias=False),
            Blur()
        )

        self.net = nn.SequentialCell(
            WeightScaledConv2d(input_channels, input_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(input_channels, filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2),
        )

        self.downsample = nn.SequentialCell(
            WeightScaledConv2d(filters, filters, 3, 2),
            Blur()
        )

    def construct(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = self.downsample(x)
        x = (x + res)  # * (1 / math.sqrt(2))
        return x


class ResUpBlock(nn.Cell):
    def __init__(self, input_channels, filters, upsample = True):
        super().__init__()
        self.upsample = upsample
        self.conv_res = WeightScaledConv2d(input_channels, filters, 1, 1, bias=False)
        self.net = nn.SequentialCell(
            WeightScaledConv2d(input_channels, input_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(input_channels, filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2),
        )

    def construct(self, x):
        b, c, h, w = x.shape
        if self.upsample:
            x = ops.interpolate(x, size=(2 * h, 2 * w), mode='bilinear', align_corners=False)
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res)  # * (1 / math.sqrt(2))
        return x
