import numpy as np

def gene_ante_deter(data, K):
    k = K
    C = var_part(data, k)
    v = C
    b = kernel_width(C, data, 1, 10)
    return v, b

def var_part(data, K):
    clusters = []
    C = np.zeros((K, data.shape[1]))
    k = 1
    while k < K:
        var_dimen = np.var(data, axis=0)
        maxvar_index = np.argmax(var_dimen)
        data_maxvar = data[:, maxvar_index]
        mean_maxvar = np.mean(data_maxvar)
        class_1 = data[data_maxvar <= mean_maxvar, :]
        class_2 = data[data_maxvar > mean_maxvar, :]
        sum_norm_1 = scatter_within(class_1)
        sum_norm_2 = scatter_within(class_2)
        max_index_class = np.argmax([sum_norm_1, sum_norm_2])
        min_index_class = np.argmin([sum_norm_1, sum_norm_2])
        data = class_2 if max_index_class == 1 else class_1
        clusters.append(class_1 if min_index_class == 1 else class_2)
        k += 1
        if k == K:
            clusters.append(data)
    for i in range(K):
        C[i, :] = np.mean(clusters[i], axis=0)
    return C

def scatter_within(data):
    mean_data = np.mean(data, axis=0)
    mean_data = np.tile(mean_data, (data.shape[0], 1))
    data_new = data - mean_data
    sum_norm = np.sum(np.linalg.norm(data_new, axis=1))
    return sum_norm

def kernel_width(C, data, a, b):
    distances = np.linalg.norm(data[:, np.newaxis, :] - C, axis=2)
    max_distance = np.max(distances)
    sigma = (a + b * max_distance) / np.sqrt(2 * C.shape[1])
    return sigma