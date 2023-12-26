import os
import mindspore as ms
from mindspore import ops, Tensor
from math import pi


def reparameterize(mu, logvar):
    sig = ops.exp(0.5 * logvar)
    sample = mu + ops.randn_like(sig) * sig
    return sample


def compute_kl_divergence(mu, logvar):
    mu = ops.flatten(mu, start_dim=2)
    logvar = ops.flatten(logvar, start_dim=2)
    kl = ops.mean(-0.5*ops.sum(logvar + 1 - mu.pow(2) - logvar.exp(), dim=-1))
    return kl


def get_indices():
    layer_list1 = [[9, 8], [3, 2], [10, 7], [4, 1]]
    layer_list2 = [[11, 9], [11, 8], [5, 3], [5, 2], [12, 10], [12, 7], [6, 4], [6, 1]]
    layer_list3 = [[13, 0], [13, 11], [13, 9], [13, 8], [13, 5], [13, 3], [13, 2], [13, 12],
    [13, 10], [13, 7], [13, 6], [13, 4], [13, 1]]
    layer_indexes = [ops.ones([15], dtype=ms.int64) * 14
                     for _ in range(3)]
    layer_lists = [layer_list1, layer_list2, layer_list3]
    for i, layer_list in enumerate(layer_lists):
        for edge in layer_list:
            layer_indexes[i][edge[1]] = edge[0]
    return layer_indexes


def pose_out_process(pose):
    scale, theta, trans = ops.split(pose, [2, 1, 2], axis=-1)
    scale = 0.8 * ops.sigmoid(scale) + 0.2
    # theta = torch.tanh(theta) * pi
    theta = theta * pi
    trans = ops.tanh(trans) * 5
    pose = ops.cat([scale, theta, trans], axis=-1)
    return pose


def accumulate_pose(layer_indices, relative_pose):
    """
    :param relative_pose: tensor [batch_size, num_parts, 4]
    :param layer_indices: list [[...], [...], [...]]
    :return: absolute_pose: tensor [batch_size, num_parts, 2, 3]
    """
    relative_scale, relative_theta, relative_trans = ops.split(
        relative_pose, [2, 1, 2], axis=-1)
    batch_size, num_parts = relative_pose.shape[:2]
    pad_scale = ops.ones([batch_size, 1, 2])
    pad_theta = ops.zeros([batch_size, 1, 1])
    pad_trans = ops.zeros([batch_size, 1, 2])
    scale = ops.cat([relative_scale, pad_scale], axis=1)
    theta = ops.cat([relative_theta, pad_theta], axis=1)
    trans = ops.cat([relative_trans, pad_trans], axis=1)
    for layer_indice in layer_indices:
        res_scale = ops.index_select(scale, 1, layer_indice)
        res_theta = ops.index_select(theta, 1, layer_indice)
        res_trans = ops.index_select(trans, 1, layer_indice)
        res_scale_mat = ops.diag_embed(res_scale)
        res_rot_mat = ops.cat(
            [ops.cos(res_theta), -ops.sin(res_theta), ops.sin(res_theta), ops.cos(res_theta)], axis=-1)
        res_rot_mat = res_rot_mat.reshape([batch_size, num_parts + 1, 2, 2])
        res_Rs_mat = res_rot_mat.value() @ res_scale_mat.value()
        relate_trans = res_Rs_mat @ trans.unsqueeze(-1)
        relate_trans = relate_trans.squeeze(-1)
        scale = scale * res_scale.value()
        theta = theta + res_theta.value()
        trans = relate_trans + res_trans.value()
    scale_mat = ops.diag_embed(scale)
    rot_mat = ops.cat([ops.cos(theta), -ops.sin(theta), ops.sin(theta), ops.cos(theta)], axis=-1)
    rot_mat = rot_mat.reshape([batch_size, num_parts+1, 2, 2])
    Rs_mat = rot_mat @ scale_mat
    affine_mat = ops.cat([Rs_mat, trans.unsqueeze(-1)], axis=-1)
    part_affine = affine_mat[:, :-1]
    bottom = Tensor([[0., 0., 1.]])
    bottom = bottom.unsqueeze(0).unsqueeze(0).repeat(batch_size, 0).repeat(num_parts, 1)
    pad_part_affine = ops.cat([part_affine, bottom], axis=-2)
    pad_part_affine = ops.inverse(pad_part_affine)
    part_affine = pad_part_affine[:, :, :2]
    return part_affine


def choose_log_dir(root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    folder_list = os.listdir(root_dir)
    if not folder_list:
        save_path = os.path.join(root_dir, 'version_0')
        return save_path
    folder_list.sort()
    prev_save_dir = folder_list[-1]
    prev_index = int(prev_save_dir[-1])
    save_index = prev_index + 1
    save_dir = os.path.join(root_dir, 'version_' + str(save_index))
    return save_dir
