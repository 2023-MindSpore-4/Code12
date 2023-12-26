import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import torch


def ms_tensor2pt_tensor(data: ms.Tensor):
    data_numpy = data.numpy()
    data_tensor = torch.tensor(data_numpy)
    return data_tensor


def torch_tensor2ms_tensor(data: torch.Tensor):
    data_numpy = data.cpu() .numpy()
    data_ms = ms.Tensor(data_numpy)
    return data_ms
