import mindspore.nn as nn
from mindspore import ops
from networks.stylegan_networks import DiscriminatorBlock


class WeightScaledHumanDiscriminator(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(3, 64, 3, 1, 'pad', 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 'pad', 1),
            nn.LeakyReLU(0.2),
        )
        self.discriminator_block1 = DiscriminatorBlock(64, 128)
        self.discriminator_block2 = DiscriminatorBlock(128, 256)
        self.discriminator_block3 = DiscriminatorBlock(256, 512)
        self.discriminator_block4 = DiscriminatorBlock(512, 512)
        self.fc = nn.SequentialCell(
            # WeightScaleLinear(512 * 8 * 4, 128),
            nn.Dense(512 * 8 * 4, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            # WeightScaleLinear(128, 1)
            nn.Dense(128, 1)
        )

    def construct(self, input_tensor):
        # input_tensor = add_noise(input_tensor, 0.1)
        feat = self.conv(input_tensor)
        feat = self.discriminator_block1(feat)
        feat = self.discriminator_block2(feat)
        feat = self.discriminator_block3(feat)
        feat = self.discriminator_block4(feat)
        feat = ops.flatten(feat, start_dim=1)
        output = self.fc(feat)
        return output


if __name__ == "__main__":
    human_syn = ops.randn([10, 3, 128, 64])
    d = WeightScaledHumanDiscriminator()
    score = d(human_syn)
    print(score.shape)
