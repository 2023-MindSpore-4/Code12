from mindspore import ops
import mindspore.nn as nn
from networks.weight_scaled_networks import WeightScaledLinear, WeightScaledConv2d
from networks.stylegan_networks import ResDownBlock, GeneratorBlock, ResBlock, ResUpBlock
from utils.geometric_transform import make_grid
from utils.util import accumulate_pose, get_indices, pose_out_process


class HumanPartDecoder(nn.Cell):
    def __init__(self, relation_list=None, num_parts=14, app_dim=32, deform_dim=16, pose_dim=16, img_size=(128, 64)):
        super().__init__()
        self.num_parts = num_parts
        self.img_size = img_size
        self.layer_indices = get_indices()
        base_grid = make_grid(16, 8)
        self.base_grid = base_grid[None]
        app_fc_list = []
        app_conv_list = []
        for i in range(num_parts):
            fc = nn.SequentialCell(
                WeightScaledLinear(app_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
                WeightScaledLinear(64, 128 * 8 * 4),
                nn.BatchNorm1d(128 * 8 * 4),
                nn.LeakyReLU(0.2),
            )
            conv = ResUpBlock(128, 512)
            fc.update_parameters_name('part_app_fc_{}'.format(i))
            conv.update_parameters_name('part_app_conv_{}'.format(i))
            app_fc_list.append(fc)
            app_conv_list.append(conv)
        self.app_fc_list = app_fc_list
        self.app_conv_list = app_conv_list
        deform_fc_list = []
        for i in range(num_parts):
            fc = nn.SequentialCell(
                WeightScaledLinear(deform_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
                WeightScaledLinear(64, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
                WeightScaledLinear(64, 16 * 8 * 2),
                nn.Tanh()
            )
            fc.update_parameters_name('part_deform_fc_{}'.format(i))
            deform_fc_list.append(fc)
        self.deform_fc_list = deform_fc_list
        pose_fc_list = []
        for i in range(num_parts):
            fc = nn.SequentialCell(
                WeightScaledLinear(pose_dim, 64),
                # nn.Linear(pose_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
                WeightScaledLinear(64, 256),
                # nn.Linear(64, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                WeightScaledLinear(256, 256),
                # nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                WeightScaledLinear(256, 64),
                # nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
                WeightScaledLinear(64, 5, weight_init='zero'),
                # nn.Linear(64, 5)
            )
            fc.update_parameters_name('part_pose_fc_{}'.format(i))
            pose_fc_list.append(fc)
        self.pose_fc_list = pose_fc_list
        self.generator_block1 = GeneratorBlock(512, 256, upsample=False, upsample_rgb=False)
        self.generator_block2 = GeneratorBlock(256, 128, upsample=False, upsample_rgb=False)  # [64, 32]
        self.generator_block3 = GeneratorBlock(128, 64, upsample=False, upsample_rgb=False)  # [128, 64]

    def construct(self, z_app, z_deform, z_pose):
        batch_size = z_app.shape[0]
        app_latent = self.process_app(z_app)
        app_latent = app_latent.reshape([batch_size * self.num_parts, *app_latent.shape[2:]])
        res_deform_field = self.process_deform_field(z_deform)
        res_deform_field = res_deform_field.reshape([batch_size * self.num_parts, 16, 8, 2])
        deform_field = self.base_grid + res_deform_field
        deformed_latent = ops.grid_sample(app_latent, deform_field, mode='bilinear', align_corners=False)
        relative_pose = self.process_pose(z_pose)
        relative_pose = pose_out_process(relative_pose)
        pose = accumulate_pose(self.layer_indices, relative_pose)
        pose = pose.reshape([batch_size * self.num_parts, 2, 3])
        pose_grid = ops.affine_grid(pose, size=(batch_size * self.num_parts, 512, 128, 64), align_corners=False)
        deformed_latent = ops.grid_sample(deformed_latent, pose_grid, mode='bilinear', align_corners=False)
        deformed_latent = deformed_latent.reshape([batch_size, self.num_parts, 512, 128, 64])
        deformed_latent = ops.sum(deformed_latent, dim=1)
        x, rgb = self.generator_block1(deformed_latent)
        x, rgb = self.generator_block2(x, rgb)
        _, img_syn = self.generator_block3(x, rgb)
        white_rect_mask = ops.ones([batch_size * self.num_parts, 1, 16, 8])
        deformed_mask = ops.grid_sample(white_rect_mask, deform_field, mode='bilinear', align_corners=False)
        mask_pose_grid = ops.affine_grid(pose, size=(batch_size * self.num_parts, 1, 128, 64), align_corners=False)
        transformed_deformed_mask = ops.grid_sample(deformed_mask, mask_pose_grid, mode='bilinear', align_corners=False)
        transformed_deformed_mask = transformed_deformed_mask.reshape([batch_size, self.num_parts, 1, *self.img_size])
        transformed_mask = ops.grid_sample(white_rect_mask, mask_pose_grid, mode='bilinear', align_corners=False)
        transformed_mask = transformed_mask.reshape([batch_size, self.num_parts, 1, *self.img_size])
        return img_syn, transformed_mask, transformed_deformed_mask, res_deform_field, relative_pose

    def process_app(self, z):
        app_list = []
        for i, fc in enumerate(self.app_fc_list):
            conv = self.app_conv_list[i]
            app_i = fc(z[:, i])
            app_i = app_i.reshape(-1, 128, 8, 4)
            app_i = conv(app_i) # [batch_size, 512, 16, 8]
            app_list.append(app_i)
        app = ops.stack(app_list, axis=1)
        return app

    def process_deform_field(self, z):
        deform_field_list = []
        for i, fc in enumerate(self.deform_fc_list):
            deform_field_i = fc(z[:, i])
            deform_field_i = deform_field_i.reshape(-1, 8, 4, 2)
            deform_field_i = deform_field_i * 0.3  # + self.base_grid
            deform_field_list.append(deform_field_i)
        deform_field = ops.stack(deform_field_list, axis=1)
        return deform_field

    def process_pose(self, z):
        pose_list = []
        for i, fc in enumerate(self.pose_fc_list):
            pose_i = fc(z[:, i])
            pose_list.append(pose_i)
        pose = ops.stack(pose_list, axis=1)
        return pose


if __name__ == "__main__":
    z_app = ops.randn([10, 14, 32])
    z_deform = ops.randn([10, 14, 16])
    pose = ops.randn([10, 14, 16])
    relation = [
        [[13, 0]],
        [[13, 11], [11, 9], [9, 8]],
        [[13, 5], [5, 3], [3, 2]],
        [[13, 12], [12, 10], [10, 7]],
        [[13, 6], [6, 4], [4, 1]]
    ]
    decoder = HumanPartDecoder(relation, 14, 32)
    o1, o2, o3, o4, o5 = decoder(z_app, z_deform, pose)
    print(o1.shape)
    print(o2.shape)
    print(o3.shape)
    print(o4.shape)
    print(o5.shape)
