import mindspore
from mindspore import Tensor, ops


def box_filter(src: Tensor, r) -> Tensor:
    N, C, H, W = src.shape

    weight = 1/((2*r+1)**2)

    kernel = ops.ones((C, 1, 2*r+1, 2*r+1), mindspore.float32)*weight

    return ops.function.conv2d(src, kernel, padding=r, group=C, pad_mode="pad")


def guided_filter(src: Tensor, guided: Tensor, r, eps=0.01) -> Tensor:
    N, C, H, W = src.shape

    n = box_filter(ops.ones((1, 1, H, W), mindspore.float32), r)

    mean_x = box_filter(src, r)/n
    mean_y = box_filter(guided, r)/n

    cov_xy = box_filter(src*guided, r)/n-mean_x*mean_y
    var_x = box_filter(src*src, r)/n-mean_x*mean_x

    a=cov_xy/(var_x+eps)
    b=mean_y-a*mean_x

    mean_a=box_filter(a,r)/n
    mean_b=box_filter(b,r)/n

    out=mean_a*src+mean_b

    return out
