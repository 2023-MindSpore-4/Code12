import mindspore as ms
from mindspore import Tensor, ops
import mindspore.nn as nn
from networks.weight_scaled_networks import WeightScaledConv2d
from networks.batch_gcn import BatchGCN
from networks.stylegan_networks import GeneratorBlock
from utils.geometric_transform import geometric_transform1, get_center_from_mask


class WeightScaledGNNDeformGenerator(nn.Cell):
    """
    mask deformed human generator
    """
    def __init__(self, edges=None, num_parts=6, z_dim=64, img_size=(128, 64)):
        super().__init__()
        self.num_parts = num_parts
        self.edges = edges
        self.z_dim = z_dim
        self.img_size = img_size
        part_fc_list = []
        for _ in range(num_parts):
            fc = nn.SequentialCell(
                # WeightScaleLinear(z_dim, 512),
                nn.Dense(z_dim, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                # WeightScaleLinear(512, 512 * 8 * 4),
                nn.Dense(512, 512 * 8 * 4),
                nn.BatchNorm1d(512 * 8 * 4),
                nn.Tanh(),
            )
            part_fc_list.append(fc)
        self.part_fc_list = part_fc_list
        self.gnn = BatchGCN(edges=edges, node_num=self.num_parts)
        self.get_pose_from_mask = nn.SequentialCell(
            nn.Conv2d(1, 64, 3, 1, 'pad', 1, has_bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 256, 3, 1, 'pad', 1, has_bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 64, 3, 1, 'pad', 1, has_bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 16, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 4, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.conv = nn.SequentialCell(
            WeightScaledConv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(512, 512, 3, 2, 1),
        )
        self.generator_block1 = GeneratorBlock(512, 256)  # [32, 16]
        self.generator_block2 = GeneratorBlock(256, 128)  # [64, 32]
        self.generator_block3 = GeneratorBlock(128, 64, upsample_rgb=False)  # [128, 64]

    def construct(self, part_masks, z):
        # part masks [batch_size, num_parts, H, W] ; z [batch_size, num_parts, feat_dim]
        batch_size = part_masks.shape[0]
        part_center = get_center_from_mask(part_masks)  # [batch_size, num_parts, 2]
        part_center = part_center.reshape([-1, 2])
        part_poses = self.get_pose_from_mask(part_masks.reshape([-1, 1, *self.img_size]))
        part_Rs = geometric_transform1(part_poses.squeeze(), similarity=True, as_matrix=True)  # [batch_size, num_parts, 2, 2]
        # part_deform_filed = part_deform_filed.reshape([batch_size, self.num_parts, -1, *self.img_size])
        part_translate = - part_Rs @ part_center.unsqueeze(-1)
        part_poses = ops.cat([part_Rs, part_translate], axis=-1)
        joint_z = self.gnn(z)
        processed_z = self.process_z(joint_z)
        processed_z = processed_z.reshape([batch_size, self.num_parts, -1, 8, 4])
        latent = processed_z.reshape([batch_size * self.num_parts, -1, 8, 4])
        part_deform_filed = ops.affine_grid(part_poses, size=(batch_size * self.num_parts, 512, 32, 16), align_corners=True)
        deformed_latent = ops.grid_sample(latent, part_deform_filed, mode='bilinear', align_corners=True)
        deformed_latent = deformed_latent.reshape([batch_size, self.num_parts, 512, 32, 16])
        mask_attention = ops.interpolate(part_masks, size=[32, 16])
        mask_attention = 0.9 * mask_attention + 0.1
        deformed_latent = deformed_latent * mask_attention[:, :, None]
        deformed_latent = ops.sum(deformed_latent, dim=1)
        # img_syn = self.backbone_generator(deformed_latent)
        x = self.conv(deformed_latent)
        x, rgb = self.generator_block1(x)
        x, rgb = self.generator_block2(x, rgb)
        _, img_syn = self.generator_block3(x, rgb)
        mask_deform_field = ops.affine_grid(part_poses, size=(batch_size * self.num_parts, 1, 32, 16), align_corners=True)
        white_mask = ops.ones([batch_size * self.num_parts, 1, 32, 16])
        deformed_mask = ops.grid_sample(white_mask, mask_deform_field, align_corners=True)
        deformed_mask = ops.interpolate(deformed_mask, size=[128, 64])
        deformed_mask = deformed_mask.reshape([batch_size, self.num_parts, 1, *self.img_size])
        return img_syn, deformed_mask

    def process_z(self, z):
        processed_z_list = []
        for i, fc in enumerate(self.part_fc_list):
            processed_z_i = fc(z[:, i])
            processed_z_list.append(processed_z_i)
        processed_z = ops.stack(processed_z_list, axis=1)
        return processed_z


if __name__ == "__main__":
    masks = ops.randn([10, 6, 128, 64])
    z = ops.randn([10, 6, 64])
    edges = [[5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5]]
    generator = WeightScaledGNNDeformGenerator(edges=edges)
    img, part_masks = generator(masks, z)
    print(img.shape)
    print(part_masks.shape)
