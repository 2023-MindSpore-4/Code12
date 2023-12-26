import mindspore.nn as nn
from mindspore.ops import permute, reshape
from mindspore import ops, Parameter


class GeoNet(nn.Cell):
    def __init__(self, zg_dim=64, nchan=3, imgsize=64, batchsize=9, kersize=[3,3,5,5], chb=16, fmd=[4,8,16,32,64], stride=[1, 2, 2, 1]):
        super(GeoNet, self).__init__()
        self.zg_dim = zg_dim
        self.nchan = nchan
        self.imgsize = imgsize
        self.bs = batchsize
        self.kersize = kersize
        self.chb = chb
        self.channelg = [chb * 8, chb * 4, chb * 2, chb * 1]
        self.fmd = fmd
        self.name = 'gengeo'
        self.base_geo = self.init_geo()
        self.dense = nn.SequentialCell(
            nn.Dense(self.zg_dim, self.fmd[0] * self.fmd[0] * self.channelg[0]),
            nn.ReLU()
        )
        self.clip = nn.Hardtanh(-1, 1)
        self.deconv2d = nn.SequentialCell(
            nn.Conv2d(self.channelg[0], self.channelg[1], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channelg[1], self.channelg[1], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(self.fmd[1], self.fmd[1])),
            nn.Conv2d(self.channelg[1], self.channelg[1], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channelg[1], self.channelg[1], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(self.fmd[2], self.fmd[2])),
            nn.Conv2d(self.channelg[1], self.channelg[2], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channelg[2], self.channelg[2], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(self.fmd[3], self.fmd[3])),
            nn.Conv2d(self.channelg[2], self.channelg[3], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channelg[3], self.channelg[3], 3, 1, "pad", 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(self.fmd[4], self.fmd[4])),
            nn.Conv2d(self.channelg[3], 2, 3, 1, "pad", 1),
            # nn.Sigmoid(),
            nn.Tanh()
        )

    def init_geo(self):
        x = ops.linspace(-1, 1, self.imgsize)
        y = ops.linspace(-1, 1, self.imgsize)
        xx, yy = ops.meshgrid(x, y, indexing='ij')
        grid = ops.stack([yy, xx], axis=0)
        grid = ops.unsqueeze(grid, dim=0)
        return grid

    def construct(self, z):
        bs = z.shape[0]
        hc = self.dense(z)
        hc = reshape(hc, [bs, self.channelg[0], self.fmd[0], self.fmd[0]])
        gdf = self.deconv2d(hc)
        gdf = gdf + self.base_geo
        gdf = permute(gdf, (0, 2, 3, 1))
        return gdf


class SmoothGrid(nn.Cell):
    def __init__(self, kersize, std):
        super(SmoothGrid, self).__init__()
        self.kersize = kersize
        filterx = self.gaussian_fn(kersize, std).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        filtery = self.gaussian_fn(kersize, std).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        self.filterx = Parameter(filterx, requires_grad=False)
        self.filtery = Parameter(filtery, requires_grad=False)

    def gaussian_fn(self, M, std):
        n = ops.arange(0, M) - (M - 1.0) / 2.0
        sig2 = 2 * std * std
        w = ops.exp(-n ** 2 / sig2)
        return w

    def construct(self, gridfiled):
        _,_, w, h = gridfiled.shape
        fullx = ops.conv2d(gridfiled[:, 0, :, :].unsqueeze(1), self.filterx, pad_mode='pad',
                           padding=(0, int((self.kersize - 1) / 2)))
        fully = ops.conv2d(gridfiled[:, 1, :, :].unsqueeze(1), self.filtery, pad_mode='pad',
                           padding=(int((self.kersize - 1) / 2), 0))
        output_grid = ops.cat((fullx[:, :, 0:w, 0:w], fully[:, :, 0:w, 0:w]), 1)
        return output_grid


if __name__ == "__main__":
    z = ops.randn([2, 64])
    d = GeoNet()
    grid = d(z)
    print(grid.shape)
