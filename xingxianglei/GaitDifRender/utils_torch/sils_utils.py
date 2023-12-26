import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
import numpy as np
from utils_torch import image_processing

from torchvision.utils import make_grid
import mindspore as ms
import mindspore.ops as ops

def cut_silhouette(sils_batch):
    n, h, w = sils_batch.shape
    # Get the upper and lower points
    y_sum = sils_batch.sum(axis=2)
    y_top = (y_sum > 0.9).float().argmax(axis=1)
    y_btm = (y_sum > 0.9).cumsum(axis=1).argmax(axis=1)

    # 开始对rgbs进行数据预处理，使用ratios对rgbs进行resize
    theta = ms.Tensor([
        [1.0, 0, 0],
        [0, 1.0, 0]],dtype=ms.float32).tile((n, 1, 1))
    theta[:, 1, 1] = 1.0*(y_btm - y_top) / h
    theta[:, 0, 0] = 0.5

    # theta[:, 1, 2] = (2 * y_top / h) *  (y_btm - y_top) / h
    # theta[:, 0, 2] = -y_top / (y_btm - y_top) * h
    sils_batch = sils_batch.unsqueeze(1)
    grid = ops.affine_grid(theta, sils_batch.shape)
    sils_batch = ops.grid_sample(sils_batch, grid)
    sils_batch = ops.interpolate(sils_batch, size=(64, 44), mode='bilinear')
    return sils_batch
