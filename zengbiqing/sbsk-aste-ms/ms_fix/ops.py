from mindspore import Tensor
import mindspore as ms
def numel(x: Tensor):
    s = x.shape
    if len(s) == 0:
        return 0

    v = 1
    for dim in s:
        v *= dim
    return v