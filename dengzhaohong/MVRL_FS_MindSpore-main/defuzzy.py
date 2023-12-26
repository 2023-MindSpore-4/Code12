import mindspore as ms
import numpy as np

def defuzzy(num, N, cluster):
    num = num.transpose()
    num_max = np.max(num)
    de_U = np.zeros((cluster, N))

    for i in range(num.shape[0]):
        for j in range(1, num_max + 1):
            if num[i] == j:
                de_U[j - 1, i] = 1

    return ms.Tensor(de_U, dtype=ms.float32)