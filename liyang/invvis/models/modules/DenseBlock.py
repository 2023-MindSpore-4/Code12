from mindspore import nn, ops

class DenseBlock(nn.Cell):
    def __init__(self, channel_in, channel_out, gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, has_bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, has_bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, has_bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, has_bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, has_bias=bias)
        self.lrelu = nn.LeakyReLU(alpha=0.2)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(ops.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(ops.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(ops.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(ops.cat((x, x1, x2, x3, x4), 1))
        return x5

