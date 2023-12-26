from mindspore import Tensor

def gray_ish(t:Tensor)->Tensor:
    N,C,H,W=t.shape
    ret=t[:,0]*0.299+t[:,1]*0.587+t[:,2]*0.114
    return ret.reshape(N,1,H,W)