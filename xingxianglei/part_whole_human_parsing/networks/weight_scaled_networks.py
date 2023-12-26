from mindspore import ops, Parameter
import mindspore.nn as nn
from math import sqrt


class WeightScaledConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.scale = 1 / sqrt(in_channels * kernel_size * kernel_size)
        self.weight = Parameter(ops.randn([out_channels, in_channels, kernel_size, kernel_size]))
        if bias:
            self.bias = Parameter(ops.zeros([out_channels]))
        else:
            self.bias = None

    def construct(self, input_tensor):
        output = ops.conv2d(input_tensor, self.weight * self.scale, self.bias, stride=self.stride, pad_mode='pad', padding=self.padding)
        return output


class WeightScaledLinear(nn.Cell):
    def __init__(self, in_channels, out_channels, weight_init='normal', bias_init='zero'):
        super().__init__()
        self.scale = 1 / sqrt(in_channels)
        self.linear = nn.Dense(in_channels, out_channels, weight_init=weight_init, bias_init=bias_init)

    def construct(self, input_tensor):
        output = self.linear(input_tensor * self.scale)
        return output


class WeightScaledResDownBlock(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.SequentialCell(
            nn.MaxPool2d(2, 2),
            WeightScaledConv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.conv2 = nn.SequentialCell(
            WeightScaledConv2d(in_channels, out_channels, 3, 1, 1),
            WeightScaledConv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x_in):
        shortcut = self.conv1(x_in)
        x = self.conv2(x_in)
        x_out = shortcut + x
        return x_out


if __name__ == "__main__":
    a = ops.randn([3, 64])
    b = ops.randn([3, 64, 64, 64])
    linear = WeightScaledLinear(64, 16)
    conv = WeightScaledConv2d(64, 64, 3, 1, 1)
    c = linear(a)
    d = conv(b)
    print(c.shape)
    print(d.shape)
