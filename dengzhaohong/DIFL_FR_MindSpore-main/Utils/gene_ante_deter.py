import mindspore

def scatter_within(data):
    mean_data = mindspore.ops.mean(data, axis=0)
    mean_data = mean_data.repeat(data.size(0), 1)
    data_new = data - mean_data
    sum_norm = mindspore.ops.norm(data_new, axis=1).sum()
    return sum_norm

def var_part(data, K):
    clusters = []
    C = mindspore.ops.zeros(K, data.size(1))
    k = 1
    while k < K:
        var_dimen = mindspore.ops.var(data, axis=0)
        _, maxvar_index = mindspore.ops.max(var_dimen, axis=0)
        data_maxvar = data[:, maxvar_index]
        mean_maxvar = mindspore.ops.mean(data_maxvar)
        class_1 = data[data_maxvar <= mean_maxvar, :]
        class_2 = data[data_maxvar > mean_maxvar, :]
        sum_norm_1 = scatter_within(class_1)
        sum_norm_2 = scatter_within(class_2)
        _, max_index_class = mindspore.ops.max(mindspore.tensor([sum_norm_1, sum_norm_2]))
        _, min_index_class = mindspore.ops.min(mindspore.tensor([sum_norm_1, sum_norm_2]))
        data = class_1 if max_index_class == 1 else class_2
        clusters.append(class_2 if max_index_class == 1 else class_1)
        k = k + 1
        if k == K:
            clusters.append(data)
    
    for i in range(K):
        C[i, :] = mindspore.ops.mean(clusters[i], axis=0)
    
    return C

def gene_ante_deter(data, K):
    k = K
    C = var_part(data, k)
    v = C
    b = kernel_width(C, data, 1, 10)
    return v, b