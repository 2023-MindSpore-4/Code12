import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import Tensor
import numpy as np
from mindspore import dtype as mstype

class CosineMarginProduct(nn.Cell):
    # 10575 30.0
    def __init__(self, m=0.05):
        super(CosineMarginProduct, self).__init__()
        self.m = m

    def construct(self, input, label):

        one_hot = ops.zeros_like(input)

        output = (input - one_hot * self.m)
        return output

