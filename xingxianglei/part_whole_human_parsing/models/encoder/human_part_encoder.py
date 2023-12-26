from mindspore import ops
import mindspore.nn as nn
from networks.weight_scaled_networks import WeightScaledConv2d, WeightScaledLinear
from networks.stylegan_networks import ResDownBlock
from networks.batch_gcn import BatchGCN
from networks.conv_attn import ConvAttn


class HumanPartEncoder(nn.Cell):
    """
    Encode image into app_code, pose_code
    """
    def __init__(self, edges=None, num_parts=14, feat_dim=128, app_dim=64, deform_dim=32, pose_dim=32):
        super().__init__()
        self.num_parts = num_parts
        self.feat_dim = feat_dim
        self.app_dim = app_dim
        self.deform_dim = deform_dim
        self.pose_dim = pose_dim
        self.base_conv = WeightScaledConv2d(3, 64, 1)
        self.feature_extractor = nn.SequentialCell(
            ResDownBlock(64, 128),
            ResDownBlock(128, 256),
            ResDownBlock(256, 512),
        )
        self.conv_attn = ConvAttn(512, num_parts=num_parts, feat_dim=feat_dim)
        self.part_feat_gnn = BatchGCN(edges=edges, node_num=num_parts, feat_dim=feat_dim, stat_dim=feat_dim)
        self.encode_app = nn.SequentialCell(
            WeightScaledConv2d(feat_dim, feat_dim, 3, 1, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(feat_dim, feat_dim, 3, 2, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(feat_dim, app_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Tanh(),
        )
        self.encode_deform = nn.SequentialCell(
            WeightScaledConv2d(feat_dim, feat_dim, 3, 1, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(feat_dim, feat_dim, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(feat_dim, deform_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Tanh(),
        )
        self.pose_global_pool = nn.SequentialCell(
            WeightScaledConv2d(feat_dim, feat_dim, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(feat_dim, feat_dim, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.pose_feat_gnn = BatchGCN(edges=edges, node_num=num_parts, feat_dim=feat_dim, stat_dim=feat_dim)
        # self.encode_pose = WeightScaledLinear(feat_dim, pose_dim, weight_init='zero')
        self.encode_pose = nn.SequentialCell(
            WeightScaledLinear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.LeakyReLU(0.2),
            WeightScaledLinear(feat_dim, pose_dim),
            nn.Tanh()
        )

    def construct(self, img):
        batch_size = img.shape[0]
        base_feat = self.base_conv(img)
        feature = self.feature_extractor(base_feat)
        part_feat, attn = self.conv_attn(feature)
        part_feat = part_feat.reshape([batch_size * self.num_parts, *part_feat.shape[2:]])
        # encode part appearance latent code [batch_size, num_parts, app_dim, 8, 4]
        part_z_app = self.encode_app(part_feat)
        part_z_app = part_z_app.reshape(batch_size, self.num_parts, self.app_dim)
        part_z_app_mu, part_z_app_logvar = ops.split(part_z_app, self.app_dim // 2, -1)
        # encode part deform latent code [batch_size, num_parts, deform_dim, 8, 4]
        part_z_deform = self.encode_deform(part_feat)
        part_z_deform = part_z_deform.reshape([batch_size, self.num_parts, self.deform_dim])
        part_z_deform_mu, part_z_deform_logvar = ops.split(part_z_deform, self.deform_dim // 2, -1)
        # encode part pose [batch_size, num_parts, pose_dim]
        part_pose_feat = self.pose_global_pool(part_feat)
        part_pose_feat = part_pose_feat.reshape([batch_size, self.num_parts, -1])
        part_pose_feat = self.pose_feat_gnn(part_pose_feat)
        part_pose_feat = part_pose_feat.reshape([batch_size * self.num_parts, -1])
        part_z_pose = self.encode_pose(part_pose_feat)
        part_z_pose = part_z_pose.reshape([batch_size, self.num_parts, self.pose_dim])
        part_z_pose_mu, part_z_pose_logvar = ops.split(part_z_pose, self.pose_dim // 2, axis=-1)
        # part_z_pose = pose_out_process(part_z_pose)
        return (part_z_app_mu, part_z_app_logvar,
                part_z_deform_mu, part_z_deform_logvar,
                part_z_pose_mu, part_z_pose_logvar, attn)


if __name__ == "__main__":
    a = ops.randn([10, 3, 128, 64])
    edges = [[0, 13], [11, 13], [9, 11], [8, 9], [5, 13], [3, 5], [2, 3], [12, 13], [10, 12], [7, 10],
             [6, 13], [4, 6], [1, 4], [13, 0], [13, 11], [11, 9], [9, 8], [13, 5], [5, 3], [3, 2], [13, 12],
             [12, 10], [10, 7], [13, 6], [6, 4], [4, 1]]
    encoder = HumanPartEncoder(edges)
    encoder = encoder
    o1, o2, o3, o4, o5, o6, o7 = encoder(a)
    print(o1.shape)
    print(o2.shape)
    print(o3.shape)
    print(o4.shape)
    print(o5.shape)
    print(o6.shape)
    print(o7.shape)
