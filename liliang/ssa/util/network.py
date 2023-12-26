import mindspore
from mindspore import nn, Tensor, ops
from mindspore.ops.operations import image_ops
from conf import conf
from util.spectral_norm import Conv2d_Spetral_Norm

if conf["sn"]:
    conv3x3=Conv2d_Spetral_Norm
else:
    conv3x3 = nn.Conv2d

# def conv3x3(in_channel, out_channel, kernel_size=3, stride=1, padding=1):
#     if conf["sn"]:
#         return Conv2d_Spetral_Norm(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, pad_mode="pad")
#     else:
#         return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, pad_mode="pad")


class ResidualBlock(nn.Cell):

    def __init__(self, channel):
        super().__init__()

        self.net = nn.SequentialCell(
            nn.Conv2d(channel, channel, 3, stride=1,
                      padding=1, pad_mode="pad"),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, pad_mode="pad")
        )

    def construct(self, x: Tensor) -> Tensor:
        out = x
        out = self.net(out)

        out = out+x
        out = nn.LeakyReLU()(out)
        return out


class EncodingLayer(nn.Cell):

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()

        self.net = nn.SequentialCell(
            nn.Conv2d(in_channel, in_channel, 3, stride=2,
                      padding=1, pad_mode="pad"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, out_channel, 3,
                      stride=1, padding=1, pad_mode="pad"),
            nn.LeakyReLU()
        )

    def construct(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecodingLayer(nn.Cell):

    def transform(self, before: Tensor, now: Tensor, R: int = 2) -> Tensor:
        N, C, H, W = now.shape

        f = ops.transpose(before, (0, 2, 3, 1))  # N H W C
        f = ops.reshape(f, (N*H*W, 1, C))          # N*H*W

        g_ext = ops.transpose(now, (0, 2, 3, 1)) # N H W C
        g_ext = self.extract(g_ext) # N H W C*R
        g_ext = ops.reshape(g_ext, (N*H*W, C, R*R))

        f_ext = ops.reshape(f, (N*H*W, 1, C))
        z = ops.matmul(f_ext, g_ext)
        z = ops.reshape(z, (N*H*W, R*R))
        # z = ops.squeeze(z)
        # alpha = ops.softmax(z, 1)
        alpha=self.soft_max(z)
        alpha_ext = ops.reshape(alpha, (N*H*W, R*R, 1))
        r_ext = ops.matmul(g_ext, alpha_ext)
        r = ops.reshape(r_ext, (N, H, W, C))    # N H W C
        r = ops.transpose(r, (0, 3, 1, 2))      # N C H W

        return before

    def __init__(self, in_channel: int, out_channel: int, K1=3, K2=7):
        super().__init__()
        
        self.resize_bilinear = nn.ResizeBilinear()

        self.conv1 = nn.Conv2d(in_channel, in_channel,
                               K1, stride=1, padding=K1//2, pad_mode="pad")
        self.conv2 = nn.Conv2d(in_channel, out_channel,
                               K2, stride=1, padding=K2//2, pad_mode="pad")
        self.leaky_relu = nn.LeakyReLU()

        # The ExtrctImagePatches currently has issue when using larger than [1,1,2,2]
        
        self.extract = ops.operations._inner_ops.ExtractImagePatches([1, 1, 2, 2], [1, 1, 1, 1], [
                          1, 1, 1, 1], "same")
        
        self.soft_max=nn.Softmax(1)

    def construct(self, now: Tensor, before: Tensor) -> Tensor:
        N, C, H, W = now.shape

        o1 = self.resize_bilinear(now, (H*2, W*2))
        trans = self.transform(before, o1)
        # trans = o1
        out = self.conv1(before+trans)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        return out


class Generator(nn.Cell):

    def __init__(self):
        super().__init__()

        self.proc = nn.SequentialCell(
            nn.Conv2d(3, 32, 7, stride=1, padding=3, pad_mode="pad"),
            nn.LeakyReLU()
        )

        self.e1 = EncodingLayer(32, 64)
        self.e2 = EncodingLayer(64, 128)

        self.res = nn.SequentialCell(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.afres = nn.SequentialCell(
            nn.Conv2d(128, 64, 3, stride=1, padding=1, pad_mode="pad"),
            nn.LeakyReLU()
        )

        self.d1 = DecodingLayer(64, 32)
        self.d2 = DecodingLayer(32, 3, 3, 7)

        self.out = nn.Tanh()

    def construct(self, input: Tensor) -> Tensor:
        o0 = self.proc(input)  # 256, 256, 32

        o1 = self.e1(o0)  # 128, 128, 64

        o2 = self.e2(o1)  # 64, 64, 128

        o3 = self.res(o2)  # 64, 64, 128

        o4 = self.afres(o3)  # 64, 64, 64

        o5 = self.d1(o4, o1)  # 128, 128, 32

        o6 = self.d2(o5, o0)  # 256, 256, 3

        return self.out(o6)


class Discriminator(nn.Cell):

    def __init__(self, in_channel=3):
        super().__init__()

        self.net = nn.SequentialCell(
           conv3x3(in_channel, 32, 3, stride=2, padding=1, pad_mode="pad"),
           nn.LeakyReLU(),
           conv3x3(32, 32, 3, stride=1, padding=1, pad_mode="pad"),
           nn.LeakyReLU(),

           conv3x3(32, 64, 3, stride=2, padding=1, pad_mode="pad"),
           nn.LeakyReLU(),
           conv3x3(64, 64, 3, stride=1, padding=1, pad_mode="pad"),
           nn.LeakyReLU(),

           conv3x3(64, 128, 3, stride=2, padding=1, pad_mode="pad"),
           nn.LeakyReLU(),
           conv3x3(128, 128, 3, stride=1, padding=1, pad_mode="pad"),
           nn.LeakyReLU()
        )

        self.snc = None
        
        self.mean=ops.ReduceMean()
        self.den = nn.Dense(128, 1)

    def construct(self, input: Tensor) -> Tensor:
        
        a = self.net(input)

        b = self.mean(a, (2, 3))
        
        c = self.den(b)

        return c
