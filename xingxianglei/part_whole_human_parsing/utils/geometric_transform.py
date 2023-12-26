import numpy as np
import mindspore as ms
from mindspore import ops


def geometric_transform(pose_tensors, similarity=False, nonlinear=True,
                        as_matrix=False, inverse=True):
    """
    Converts parameter tensor into an affine or similarity transform.
    :param pose_tensor: [..., 6] tensor.
    :param similarity: bool.
    :param nonlinear: bool; applies nonlinearities to pose params if True.
    :param as_matrix: bool; convers the transform to a matrix if True.
    :return: [..., 3, 3] tensor if `as_matrix` else [..., 6] tensor.
    """
    trans_xs, trans_ys, scale_xs, scale_ys, thetas, shears = pose_tensors.split(1, dim=-1)

    if nonlinear:
        k = 0.5
        # TODO: use analytically computed trans rescaling or move it out of this method
        trans_xs = ops.tanh(trans_xs * k)  # 0.8 * torch.tanh(trans_xs * k) + 0.1
        trans_ys = ops.tanh(trans_ys * k)  # 0.8 * torch.tanh(trans_ys * k) + 0.1
        scale_xs = ops.sigmoid(scale_xs) * 0.25 # 0.14 * torch.sigmoid(scale_xs) + 0.1
        scale_ys = ops.sigmoid(scale_ys) * 0.25  # 0.14 * torch.sigmoid(scale_ys) + 0.1
        scale_ys = scale_ys  # * 1.5
        shears = ops.tanh(shears * 5.)
        thetas = thetas * 2. * np.pi
    else:
        scale_xs = ops.abs(scale_xs) + 1e-2
        scale_ys = ops.abs(scale_ys) + 1e-2

    cos_thetas, sin_thetas = ops.cos(thetas), ops.sin(thetas)

    if similarity:
        # scales = scale_xs
        # scales = 1.
        # print('scale', scales)
        # poses = [scales * cos_thetas, -scales * sin_thetas, trans_xs,
        #          scales * sin_thetas, scales * cos_thetas, trans_ys]
        poses = [scale_xs * cos_thetas, -scale_ys * sin_thetas, trans_xs,
                 scale_xs * sin_thetas, scale_ys * cos_thetas, trans_ys]
    else:
        poses = [
            scale_xs * cos_thetas + shears * scale_ys * sin_thetas,
            -scale_xs * sin_thetas + shears * scale_ys * cos_thetas,
            trans_xs,
            scale_ys * sin_thetas,
            scale_ys * cos_thetas,
            trans_ys
        ]
    poses = ops.cat(poses, -1)  # shape (... , 6)

    # Convert poses to 3x3 A matrix so: [y, 1] = A [x, 1]
    if as_matrix or inverse:
        poses = poses.reshape(*poses.shape[:-1], 2, 3)
        bottom_pad = ops.zeros((*poses.shape[:-2], 1, 3)).to(poses.device)
        bottom_pad[..., 2] = 1
        # shape (... , 2, 3) + shape (... , 1, 3) = shape (... , 3, 3)
        poses = ops.cat([poses, bottom_pad], axis=-2)

    if inverse:
        poses = ops.inverse(poses)
        if not as_matrix:
            poses = poses[..., :2, :]
            poses = poses.reshape(*poses.shape[:-2], 6)

    return poses


def geometric_transform1(pose_tensors, similarity=False, nonlinear=True,
                        as_matrix=False, inverse=True):
    """
    Converts parameter tensor into an affine or similarity transform.
    :param pose_tensor: [..., 6] tensor.
    :param similarity: bool.
    :param nonlinear: bool; applies nonlinearities to pose params if True.
    :param as_matrix: bool; convers the transform to a matrix if True.
    :return: [..., 2, 2] tensor if `as_matrix` else [..., 4] tensor.
    """
    scale_xs, scale_ys, thetas, shears = pose_tensors.split(1, axis=-1)

    if nonlinear:
        k = 0.5
        # TODO: use analytically computed trans rescaling or move it out of this method
        scale_xs = ops.sigmoid(scale_xs) * 0.3  # 0.14 * torch.sigmoid(scale_xs) + 0.1
        scale_ys = ops.sigmoid(scale_ys) * 0.3 # 0.14 * torch.sigmoid(scale_ys) + 0.1
        scale_ys = scale_ys  # * 1.5
        shears = ops.tanh(shears * 5.)
        thetas = thetas * 2. * np.pi
    else:
        scale_xs = ops.abs(scale_xs) + 1e-2
        scale_ys = ops.abs(scale_ys) + 1e-2

    cos_thetas, sin_thetas = ops.cos(thetas), ops.sin(thetas)

    if similarity:
        # scales = scale_xs
        # scales = 1.
        # print('scale', scales)
        # poses = [scales * cos_thetas, -scales * sin_thetas, trans_xs,
        #          scales * sin_thetas, scales * cos_thetas, trans_ys]
        poses = [scale_xs * cos_thetas, -scale_ys * sin_thetas,
                 scale_xs * sin_thetas, scale_ys * cos_thetas]
    else:
        poses = [
            scale_xs * cos_thetas + shears * scale_ys * sin_thetas,
            -scale_xs * sin_thetas + shears * scale_ys * cos_thetas,
            scale_ys * sin_thetas,
            scale_ys * cos_thetas
        ]
    poses = ops.cat(poses, -1)  # shape (... , 4)

    # Convert poses to 3x3 A matrix so: [y, 1] = A [x, 1]
    if as_matrix or inverse:
        poses = poses.reshape(*poses.shape[:-1], 2, 2)

    if inverse:
        poses = ops.inverse(poses)
        if not as_matrix:
            poses = poses[..., :2, :]
            poses = poses.reshape(*poses.shape[:-2], 4)

    return poses


def get_center_from_mask(mask):
    batch_size, n_caps, H, W = mask.shape
    y = ops.linspace(-1, 1, H)
    x = ops.linspace(-1, 1, W)
    yy, xx = ops.meshgrid(y, x, indexing='ij')
    grid = ops.stack([xx, yy], axis=-1).unsqueeze(0).unsqueeze(0)
    mask = mask / (ops.sum(mask, dim=[2, 3], keepdim=True) + 1.e-7)
    mask = mask.unsqueeze(-1)
    t = ops.sum(mask * grid, dim=[2, 3])
    return t


def make_grid(h, w):
    y = ops.linspace(-1, 1, h)
    x = ops.linspace(-1, 1, w)
    yy, xx = ops.meshgrid(y, x, indexing='ij')
    grid = ops.stack([xx, yy], axis=-1)
    return grid
