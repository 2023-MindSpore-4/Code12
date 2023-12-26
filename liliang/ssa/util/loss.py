from conf import conf
import mindspore
import numpy as np
from mindspore import nn
from mindspore import Tensor, ops
from util.vgg import vgg19
from util.effecient_segmentation_new import effecient_segmentation
from typing import Tuple

def L_motion(pred1: Tensor, pred2: Tensor, ori1: Tensor, ori2: Tensor, pred_vgg: Tensor = None, ori_vgg: Tensor = None, VGG=vgg19(True)) -> Tensor:

    def C(t1: Tensor, t2: Tensor) -> Tensor:
        N, C, H, W = t1.shape
        t1 = t1.reshape(N, C, H*W)
        t2 = t2.reshape(N, C, H*W)

        res = t1.transpose((0, 2, 1))@t2

        return res
    
    
    if pred_vgg is None:
        pred2_feature = VGG(pred2)
    else:
        pred2_feature = pred_vgg
    if ori_vgg is None:
        ori2_feature = VGG(ori2)
    else:
        ori2_feature = ori_vgg

    pred1_feature = VGG(ori1)
    ori1_feature = VGG(ori1)

    pred_map = C(pred1_feature, pred2_feature)
    ori_map = C(ori1_feature, ori2_feature)

    cosine_loss = nn.CosineEmbeddingLoss()
    loss = cosine_loss(pred_map, ori_map, ops.ones(pred_map.shape[0]))

    return loss


def L_structure(pred_seq: Tensor, pred_vgg: Tensor = None, VGG = vgg19(True)) -> Tensor:
    if pred_vgg is None:
        pred_vgg = VGG(pred_seq)

    fill_color = effecient_segmentation(pred_seq)
    fill_vgg = VGG(fill_color)

    l1loss = nn.L1Loss()
    loss = l1loss(pred_vgg, fill_vgg)

    return loss


def L_content(pred_seq: Tensor, ori_seq: Tensor, pred_vgg: Tensor = None, ori_vgg: Tensor = None, VGG = vgg19(True)) -> Tensor:

    N, C, H, W = pred_seq.shape

    if pred_vgg is None:
        pred_vgg = VGG(pred_seq)
    if ori_vgg is None:
        ori_vgg = VGG(ori_seq)

    l1loss = nn.L1Loss()
    loss = l1loss(pred_vgg, ori_vgg)/(C*H*W)

    return loss

class Loss_tv(nn.loss.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()
        
    def construct(self, pred_seq):

        N, C, H, W = pred_seq.shape

        dx = pred_seq[:, :, 1:, :]-pred_seq[:, :, :-1, :]
        dy = pred_seq[:, :, :, 1:]-pred_seq[:, :, :, :-1]

        mean = ops.reduce_mean(dx**2)+ops.reduce_mean(dy**2)
        loss = mean/(C*H*W)
        return loss

def L_tv(pred_seq: Tensor) -> Tensor:

    N, C, H, W = pred_seq.shape

    dx = pred_seq[:, :, 1:, :]-pred_seq[:, :, :-1, :]
    dy = pred_seq[:, :, :, 1:]-pred_seq[:, :, :, :-1]

    mean = ops.reduce_mean(dx**2)+ops.reduce_mean(dy**2)
    loss = mean/(C*H*W)

    return loss

class Lsgan_loss_g(nn.loss.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.mean=ops.ReduceMean()
        
    def construct(self, fake_img):
        
        return self.mean(fake_img)
        # return ops.mean((fake_img-1)**2)
    
class Lsgan_loss_d(nn.loss.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.mean=ops.ReduceMean()
        
    def construct(self, real_img, fake_img):

        return 0.5*(self.mean((real_img-1)**2)+self.mean(fake_img**2))
        # return 0.5*(ops.mean((real_img-1)**2)+ops.mean(fake_img**2))

def lsgan_loss_g(fake_img: Tensor) -> Tensor:
    return ops.mean((fake_img-1)**2)

def lsgan_loss_d(real_img: Tensor, fake_img: Tensor) -> Tensor:
    return 0.5*(ops.mean((real_img-1)**2)+ops.mean(fake_img**2))

def lsgan_loss(real: Tensor, fake: Tensor) -> Tuple[Tensor, Tensor]:

    def g(fake_img: Tensor) -> Tensor:
        return ops.mean((fake_img-1)**2)

    def d(real_img: Tensor, fake_img: Tensor) -> Tensor:
        return 0.5*(ops.mean((real_img-1)**2)+ops.mean(fake_img**2))

    return d(real, fake), g(fake)

class Gan_loss_g(nn.loss.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.mean=ops.ReduceMean()
        self.log=ops.Log()
        
    def construct(self, fake_img):

        return -self.mean(self.log(fake_img))
        # return -ops.mean(ops.log(fake_img))
    
class Gan_loss_d(nn.loss.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()

        self.mean=ops.ReduceMean()
        self.log=ops.Log()
        
    def construct(self, real_img, fake_img):

        return -self.mean(self.log(real_img)+self.log(1-fake_img))
        # return -ops.mean(ops.log(real_img)+ops.log(1-fake_img))

def gan_loss_g(fake_img: Tensor) -> Tensor:
    return -ops.mean(ops.log(fake_img))

def gan_loss_d(real_img: Tensor, fake_img: Tensor) -> Tensor:
    return -ops.mean(ops.log(real_img)+ops.log(1-fake_img))

def gan_loss(real: Tensor, fake: Tensor) -> Tuple[Tensor, Tensor]:

    def g(fake_img: Tensor) -> Tensor:
        return -ops.mean(ops.log(fake))

    def d(real_img: Tensor, fake_img: Tensor) -> Tensor:
        return -ops.mean(ops.log(real)+ops.log(1-fake))

    return d(real, fake), g(fake)