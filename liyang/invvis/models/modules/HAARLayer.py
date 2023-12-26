from mindspore import nn ,ops


class HAARLayer(nn.Cell):
    def __init__(self, channel_in):
        super(HAARLayer, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = ops.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = ops.cat([self.haar_weights] * self.channel_in, 0)


        self.haar_weights = self.haar_weights.cuda()

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = ops.conv2d(x, self.haar_weights, has_bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = ops.swapaxes(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = ops.swapaxes(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, has_bias=None, stride=2, groups=self.channel_in)