import torch
import torch.nn.functional as F
from torchtyping import TensorType


def cut_silhouette(sils_batch: TensorType["batch", "height", "width"]):
    n, h, w = sils_batch.shape
    # Get the upper and lower points
    y_sum = sils_batch.sum(axis=2)
    y_top = (y_sum > 0.9).float().argmax(axis=1)
    y_btm = (y_sum > 0.9).cumsum(axis=1).argmax(axis=1)

    # 开始对rgbs进行数据预处理，使用ratios对rgbs进行resize
    theta = torch.tensor([
        [1, 0, 0],
        [0, 1, 0]], dtype=torch.float, device=sils_batch.device).repeat(n, 1, 1)
    theta[:, 1, 1] = (y_btm - y_top) / h
    theta[:, 0, 0] = 0.5

    sils_batch = sils_batch.unsqueeze(1)
    grid = F.affine_grid(theta, sils_batch.size())
    sils_batch = F.grid_sample(sils_batch, grid)
    sils_batch = F.interpolate(sils_batch, size=(64, 44), mode='bilinear')
    return sils_batch
