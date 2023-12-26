'''
Modifed from https://github.com/Gait3D/Gait3D-Benchmark/blob/72beab994c137b902d826f4b9f9e95b107bebd78/lib/modeling/models/smplgait.py
'''
import copy

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from data.smpl_dataset import SMPLDataSet
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
import data.sampler as Samplers
import torch.utils.data as tordata
from data.collate_fn import CollateFn
from data.smpl_collate_fun import SMPLCollateFn
from data.transform import get_transform

from modules.hybrik.models.layers.smpl.SMPL import SMPL_layer

from utils_torch.render_pytorch3d import render_mesh, render_silhouette_mesh

from utils_torch import image_processing
from utils_torch.sils_utils import cut_silhouette
from utils_torch.torch_tools import freeze_model
from mindspore_utils.utils import ms_tensor2pt_tensor,torch_tensor2ms_tensor
from torchvision.utils import make_grid
from torchvision import models

import mindspore as ms
import mindspore.nn as ms_nn
import mindspore.ops as ops
#
from mindconverter import pytorch2mindspore


def show_batch_image(title, batch_imgs):
    batch_imgs = batch_imgs.detach()
    grid = make_grid(batch_imgs, nrow=8)
    grid = grid.cpu().numpy()

    image = np.array(grid, dtype=np.uint8)
    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    else:
        image = image.transpose(1, 0)
    image_processing.show_image(title, image)


class GaitRecognitionHead(nn.Module):
    def __init__(self, shape_dim, pose_dim):
        super(GaitRecognitionHead, self).__init__()
        self.shape_mlp = nn.Sequential(
            nn.Linear(shape_dim, 128),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 512),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True), )

        # Motion processing
        self.rotation_projection = nn.Linear(pose_dim, 512)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, 8, 2048),
            num_layers=6
        )
        self.gait_mlp = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.l2_norm = nn.LayerNorm(512)

    def forward(self, pred_shape, rotation_matrices):
        """
        Parameters
        ----------
        pred_shape [B, 10]
        pred_pose [B, N ,72]

        Returns
        -------

        """
        shape_embedding = self.shape_mlp(pred_shape)
        # Motion embedding
        rotation_embedding = self.rotation_projection(rotation_matrices)
        motion_embedding = self.transformer_encoder(rotation_embedding)
        motion_embedding = motion_embedding[:, 0, :]  # Get the embedding corresponding to the added token

        # Concatenate shape and motion embeddings
        gait_embedding = torch.cat((shape_embedding, motion_embedding), dim=1)

        # Gait embedding
        gait_embedding = self.gait_mlp(gait_embedding)
        # L2 normalization
        gait_embedding = self.l2_norm(gait_embedding)

        return gait_embedding


class CustomMobileNet(ms.nn.Cell):
    def __init__(self, output_dim):
        super(CustomMobileNet, self).__init__()
        # mobilenet = models.mobilenet_v2(pretrained=True)  # 使用预训练的MobileNetV2模型
        # self.mobilenet = pytorch2mindspore  # 使用预训练的MobileNetV2模型
        from mindvision.classification.models import mobilenet_v2
        mobilenet = mobilenet_v2(num_classes=2, resize=224)
        # 移除最后一层，以适应自定义的输出维度
        self.features = copy.deepcopy( mobilenet.backbone.features)
        self.avgpool = ms.nn.Dropout(p=0.2)

        self.global_avg_pool = ms.nn.AdaptiveAvgPool2d(1)
        self.fc = ms.nn.Dense(1280, output_dim)  # 根据MobileNetV2的最后一层输出通道数调整线性层的输入通道数

    def construct(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class GaitCorrectionModule(ms.nn.Cell):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = CustomMobileNet(input_dim)
        self.mlp = ms.nn.SequentialCell(
            ms.nn.Dense(1423, 512),
            ms.nn.BatchNorm1d(512),
            ms.nn.PReLU(),
            ms.nn.Dense(512, 512),
            ms.nn.BatchNorm1d(512),
            ms.nn.BatchNorm1d(512),
            ms.nn.PReLU(),
            ms.nn.Dense(512, 512),
            ms.nn.BatchNorm1d(512),
            ms.nn.ReLU())
        self.out_mlp = ms.nn.SequentialCell(
            ms.nn.Dense(512, 512),
            ms.nn.BatchNorm1d(512),
            ms.nn.PReLU(),
            ms.nn.Dense(512, 512),
            ms.nn.BatchNorm1d(512),
            ms.nn.PReLU(),
            ms.nn.Dense(512, 29 * 3 + 10 + 23 * 2, ))
        # 设置out_mlp的权重

    def init_parameters(self):
        pass

    def construct(self, images, pose_skeleton, betas, phis):
        # image: (b, 3, 128, 128)
        # pose_skeleton: (batch_size, 29, 3)
        # betas: (batch_size, 10)
        # phis: (batch_size, 23, 2)
        b, s, c, h, w = images.shape

        image = images.view(b * s, c, h, w)
        z = self.encoder(image)
        pose_skeleton = pose_skeleton.view(b * s, -1)
        betas = betas.view(b * s, -1)
        phis = phis.view(b * s, -1)
        z = ops.cat([z, pose_skeleton, betas, phis], axis=1)

        z = self.mlp(z)

        x = self.out_mlp(z)

        delta_pose_skeleton = x[:, :87]
        delta_pose_skeleton = delta_pose_skeleton.view(b * s, 29, 3)
        delta_betas = x[:, 87:87 + 10]
        delta_phis = x[:, 97:]
        delta_phis = delta_phis.view(b * s, 23, 2)

        return delta_pose_skeleton, delta_betas, delta_phis


class SMPLCalibrateGait(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def inputs_pretreament(self, inputs):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs

        typs = typs_batch
        vies = vies_batch

        labs = list2var(labs_batch).long()

        if seqs_batch is not None:
            for feature_name in seqs_batch.keys():
                try:
                    seqs_batch[feature_name] = list2var(seqs_batch[feature_name]).float()
                except:
                    seqs_batch[feature_name] = list2var(seqs_batch[feature_name]).float()
        seqL = seqL_batch

        return seqs_batch, labs, typs, vies, seqL

    def get_loader(self, data_cfg, train=True):
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        dataset = SMPLDataSet(data_cfg, train)

        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        sampler = Sampler(dataset, **vaild_args)

        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=SMPLCollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg['num_workers'])
        return loader

    def build_network(self, model_cfg):
        # Baseline
        # torch 部分的代码，改起来比较麻烦
        self.smpl_dtype = torch.float32
        bbox_3d_shape = (2000, 2000, 2000)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 1e-3
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=self.smpl_dtype
        )
        self.smpl_faces = torch.from_numpy(self.smpl.faces.astype(np.int32))

        # 网络的骨干部分
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.TP = PackSequenceWrapper(ops.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])

        # self.gait_recognition_head = GaitRecognitionHead(10, 207)
        # self.SMPL_SeparateFCs = SeparateFCs(**model_cfg['SMPL_SeparateFCs'])
        self.gaitCorrectionModule = GaitCorrectionModule(1280)
        self.load_ckpt()
    def load_ckpt(self):
        # def mindspore_params(network):
        #     ms_params = {}
        #     ms_params_name_list = []
        #     for param in network.get_parameters():
        #         name = param.name
        #         value = param.data.asnumpy()
        #         print(name, value.shape)
        #         ms_params_name_list.append(name)
        #         ms_params[name] = value
        #     import pickle
        #
        #     with open('ms_params_name_list.pkl', 'wb') as file:
        #         pickle.dump(ms_params_name_list, file)
        #     return ms_params
        #
        # ms_params = mindspore_params(self)
        # print(ms_params)
        param_dict = ms.load_checkpoint("checkpoint/gait_checkpoint.ckpt")
        param_not_load, _ = ms.load_param_into_net(self, param_dict)
        print(param_not_load)
        smpl_ckpt = torch.load('checkpoint/smpl.pth')
        self.smpl.load_state_dict(smpl_ckpt)

        # freeze_model(self.smpl)

    def init_parameters(self):
        pass
        # for m in self.modules():
        #
        #     if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
        #         nn.init.xavier_uniform_(m.weight.data)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias.data, 0.0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight.data)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias.data, 0.0)
        #     elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
        #         if m.affine:
        #             nn.init.normal_(m.weight.data, 1.0, 0.02)
        #             nn.init.constant_(m.bias.data, 0.0)
        # self.gaitCorrectionModule.init_parameters()

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        n, s, c, h, w = ipts['rgbs'].shape
        aligned_sils = ipts['aligned-sils']

        rgbs = ipts['rgbs'].view(n * s, c, h, w)
        from equal_data.equal_data import equal

        ratios = ipts['ratios'].view(n * s, 1)
        original_sils = ipts['sils'].view(n * s, 1, 128, 128)
        # 开始对rgbs进行数据预处理，使用ratios对rgbs进行resize
        theta = ms.Tensor([
            [1, 0, 0],
            [0, 1, 0]], dtype=ms.float32).tile((original_sils.shape[0], 1, 1))

        theta[:, 0, 0] = 1 / ratios.squeeze()
        grid = ops.affine_grid(theta, original_sils.shape)
        rgbs = ops.grid_sample(rgbs, grid)
        rgbs = rgbs.view(n * s, c, 128, 128)

        rgbs = ops.interpolate(rgbs, size=(64, 64), mode='bilinear')
        rgbs = rgbs.view(n, s, c, 64, 64)

        # show_batch_image("rgbs", rgbs[0])

        original_sils = ops.grid_sample(original_sils, grid)
        original_sils = original_sils.view(n * s, 1, 128, 128)

        pred_xyz_jts_29 = ipts['pred_xyz_29']
        pred_shape = ipts['pred_betas']
        pred_phi = ipts['pred_phi']
        pred_xyz_jts_29 = pred_xyz_jts_29.view(n * s, 29, 3)
        pred_shape = pred_shape.view(n * s, -1)
        pred_phi = pred_phi.view(n * s, 23, 2)

        # 步态矫正模块
        delta_pose_skeleton, delta_betas, delta_phis = self.gaitCorrectionModule(rgbs, pred_xyz_jts_29, pred_shape,
                                                                                 pred_phi)

        pred_xyz_jts_29 = pred_xyz_jts_29 + delta_pose_skeleton
        # pred_shape = pred_shape + delta_betas
        pred_phi = pred_phi + delta_phis
        # 该部分代码用pytorch代码
        pred_xyz_jts_29_torch = ms_tensor2pt_tensor(pred_xyz_jts_29)
        pred_shape_torch = ms_tensor2pt_tensor(pred_shape)
        pred_phi_torch = ms_tensor2pt_tensor(pred_phi)
        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz_jts_29_torch.type(self.smpl_dtype) * self.depth_factor,  # unit: meter
            betas=pred_shape_torch.type(self.smpl_dtype),
            phis=pred_phi_torch.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True,
            naive=True
        )

        verts_batch = output.vertices.view(n * s, -1, 3)  # [n * s, 6890, 3]
        verts_batch = torch_tensor2ms_tensor(verts_batch)
        del output
        transl_batch = ipts['transl'].view(n * s, 3)

        # 该部分代码用pytorch3d
        focal = 1000.0
        focal = focal / 256 * 64
        smpl_faces = self.smpl_faces.type(self.smpl_dtype).cuda()
        verts_batch_torch = ms_tensor2pt_tensor(verts_batch).cuda()
        transl_batch_torch = ms_tensor2pt_tensor(transl_batch).cuda()

        with torch.cuda.amp.autocast(enabled=False):
            render_sils = render_silhouette_mesh(
                vertices=verts_batch_torch, faces=smpl_faces,
                translation=transl_batch_torch,
                focal_length=focal, height=64, width=64)
        render_sils = torch_tensor2ms_tensor(render_sils)
        # resize 128 128
        render_sils = render_sils.view(n * s, 1, 64, 64)
        render_sils = ops.interpolate(render_sils, size=(128, 128), mode='bilinear')
        render_sils = render_sils.view(n * s, 128, 128)

        cut_sils_batch = cut_silhouette(render_sils)
        cut_sils_batch = cut_sils_batch.view(n, 1, s, 64, 44)
        original_sils_batch = aligned_sils.view(n, 1, s, 64, 44)
        original_sils_batch = original_sils_batch / 255.0
        concatenated_sils_batch = cut_sils_batch
        concatenated_sils_labs = labs
        # concatenated_sils_batch = torch.cat((cut_sils_batch, original_sils_batch), dim=0)
        # concatenated_sils_labs = torch.cat((labs, labs), dim=0)
        outs = self.Backbone(concatenated_sils_batch)  # [n, c, s, h, w]
        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"axis": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': concatenated_sils_labs},
                'softmax': {'logits': logits, 'labels': concatenated_sils_labs},
                'mse': {'input': render_sils, 'target': original_sils.view(-1, 128, 128)},
                'fix_loss': {'delta_pose_skeleton': delta_pose_skeleton.view(n, s, 29, 3),
                             'delta_betas': delta_betas.view(n, s, 10),
                             'delta_phis': delta_phis.view(n, s, 23, 2)}

            },
            'visual_summary': {
                'image/sils': aligned_sils.view(n * s, 1, 64, 44),
                'image/original_sils': original_sils.view(n * s, 1, 128, 128),
                'image/render_sils': render_sils.view(n * s, 1, 128, 128),

            },
            'inference_feat': {
                'embeddings': embed[0:n]

            }
        }
        # print("SMPLCalibrateGait forward")
        return retval
