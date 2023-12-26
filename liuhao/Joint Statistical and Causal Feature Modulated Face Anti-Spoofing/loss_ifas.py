# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 5, 6, 7"
import os
import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore import Tensor


# , device
def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)

    # .to(device)

    kernel_filter = Tensor.from_numpy(kernel_filter).float()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).broadcast_to((input.shape[0], 8, input.shape[1],input.shape[2]))
    
    contrast_depth = ops.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class ContrastDepthLoss(nn.Cell):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    # , device
    def __init__(self):
        super(ContrastDepthLoss, self).__init__()
        # dongxin add 0419
        # self.device = device


    def construct(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        # , device=self.device
        contrast_out = contrast_depth_conv(out)
        # , device = self.device
        contrast_label = contrast_depth_conv(label)

        criterion_MSE = nn.MSELoss()
    
        loss = criterion_MSE(contrast_out, contrast_label)
    
        return loss


class DepthLoss(nn.Cell):
    # , device
    def __init__(self, w):
        super(DepthLoss, self).__init__()
        self.criterion_absolute_loss = nn.MSELoss()
        # device=device
        self.criterion_contrastive_loss = ContrastDepthLoss()
        self.w = w


    def construct(self, predicted_depth_map, gt_depth_map):
        absolute_loss = self.criterion_absolute_loss(predicted_depth_map, gt_depth_map)
        contrastive_loss = self.criterion_contrastive_loss(predicted_depth_map, gt_depth_map)
        return absolute_loss + self.w * contrastive_loss
