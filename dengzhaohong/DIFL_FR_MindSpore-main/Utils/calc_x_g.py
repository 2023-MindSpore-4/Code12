import mindspore
import numpy as np

def calc_x_g(x, v, b):
    # x: the original data -- n_examples * n_features
    # v: clustering centers of the fuzzy rule base -- k * n_features
    # b: kernel width of the corresponding centers of the fuzzy rule base
    # x_g: data in the new fuzzy feature space -- n_examples * (n_features+1)k

    n_examples = x.size(0)
    x_e = mindspore.ops.cat((x, mindspore.ops.ones(n_examples, 1)), axis=1)
    k, d = v.size()  # k: number of rules of TSK; d: number of dimensions

    wt = mindspore.ops.zeros(n_examples, k)
    for i in range(k):
        v1 = v[i].repeat(n_examples, 1)
        bb = b[i].repeat(n_examples, 1)
        wt[:, i] = mindspore.ops.exp(-mindspore.ops.sum((x - v1) ** 2 / bb, axis=1))
        
    wt2 = mindspore.ops.sum(wt, axis=1)

    # To avoid the situation that zeros are exist in the matrix wt2
    ss = wt2 == 0
    wt2[ss] = np.finfo(mindspore.float32).eps
    wt = wt / wt2.view(-1, 1)

    x_g = mindspore.ops.zeros(n_examples, k * (x.size(1) + 1))
    for i in range(k):
        wt1 = wt[:, i]
        wt2 = wt1.view(-1, 1).repeat(1, d + 1)
        x_g[:, i * (x.size(1) + 1):(i + 1) * (x.size(1) + 1)] = x_e * wt2
    return x_g