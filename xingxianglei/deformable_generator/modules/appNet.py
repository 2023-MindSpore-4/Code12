import mindspore.nn as nn


class AppNet(nn.Cell):
    def __init__(self, za_dim=64, nchan=3, imgsize=64, batchsize=9, kersize=[3,3,5,5], chb=16, alpha=5/8, fmd=[4,8,16,32,64], stride=[1, 2, 2, 1]):
        super(AppNet, self).__init__()
        self.za_dim = za_dim
        self.nchan = nchan
        self.imgsize = imgsize
        self.bs = batchsize
        self.kersize = kersize
        self.chb = chb
        channelg = [chb * 8, chb * 4, chb * 2, chb * 1]
        self.alpha = alpha
        self.channela = [int(i * alpha) for i in channelg]
        self.fmd = fmd
        self.name = 'genapp'
        self.dense = nn.SequentialCell(
            nn.Dense(self.za_dim, self.fmd[0] * self.fmd[0] * self.channela[0]),
            nn.ReLU()
        )
        # self.deconv2d = nn.SequentialCell(
        #     nn.Conv2dTranspose(self.channela[0], self.channela[1], kersize[0], stride[0]),
        #     nn.ReLU(),
        #     nn.Conv2dTranspose(self.channela[1], self.channela[2], kersize[1], stride[1]),
        #     nn.ReLU(),
        #     nn.Conv2dTranspose(self.channela[2], self.channela[3], kersize[2], stride[2]),
        #     nn.ReLU(),
        #     nn.Conv2dTranspose(self.channela[3], 3, kersize[3], stride[3]),
        #     nn.Sigmoid(),
        #     nn.Upsample(size=(self.imgsize, self.imgsize), mode='bilinear'),
        # )
        self.deconv2d = nn.SequentialCell(
            nn.Conv2d(self.channela[0], self.channela[1], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channela[1], self.channela[1], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(self.fmd[1], self.fmd[1]), mode='bilinear'),
            nn.Conv2d(self.channela[1], self.channela[1], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channela[1], self.channela[1], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(self.fmd[2], self.fmd[2]), mode='bilinear'),
            nn.Conv2d(self.channela[1], self.channela[2], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channela[2], self.channela[2], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(self.fmd[3], self.fmd[3]), mode='bilinear'),
            nn.Conv2d(self.channela[2], self.channela[3], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channela[3], self.channela[3], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(self.fmd[4], self.fmd[4]), mode='bilinear'),
            nn.Conv2d(self.channela[3], 3, 3, 1, "pad", 1),
            nn.Sigmoid(),
        )

    def construct(self, z):
        bs = z.shape[0]
        hc = self.dense(z)
        hc = hc.reshape([bs, self.channela[0], self.fmd[0], self.fmd[0]])
        gx = self.deconv2d(hc)
        return gx
