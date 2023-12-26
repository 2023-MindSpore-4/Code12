import mindspore as ms
import numpy as np

def EuDist2(fea_a, fea_b=None, bSqrt=True):
    if fea_b is None:
        aa = ms.sum(fea_a * fea_a, axis=1)
        ab = fea_a @ fea_a.transpose()

        if isinstance(aa, ms.sparse_tensor.SparseTensor):
            aa = ms.sparse_tensor.to_dense(aa)

        D = ms.add(ms.expand_dims(aa, 1), ms.expand_dims(aa, 0)) - 2 * ab
        D = ms.maximum(D, 0)
        if bSqrt:
            D = ms.sqrt(D)
        D = ms.maximum(D, D.transpose())
    else:
        aa = ms.sum(fea_a * fea_a, axis=1)
        bb = ms.sum(fea_b * fea_b, axis=1)
        ab = fea_a @ fea_b.transpose()

        if isinstance(aa, ms.sparse_tensor.SparseTensor):
            aa = ms.sparse_tensor.to_dense(aa)
            bb = ms.sparse_tensor.to_dense(bb)

        D = ms.add(ms.expand_dims(aa, 1), ms.expand_dims(bb, 0)) - 2 * ab
        D = ms.maximum(D, 0)
        if bSqrt:
            D = ms.sqrt(D)

    return D