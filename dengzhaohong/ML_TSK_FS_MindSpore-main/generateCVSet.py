import mindspore

def generateCVSet(X, Y, kk, index, totalCV):
    assert index <= 10
    assert totalCV <= 10
    m = X.size(0)
    slice_size = int(mindspore.ops.ceil(m / totalCV))
    start_index = (index - 1) * slice_size
    end_index = min(index * slice_size, m)

    test_x = X[kk[start_index:end_index], :]
    test_y = Y[kk[start_index:end_index], :]

    train_indices = mindspore.ops.cat((kk[:start_index], kk[end_index:]))
    train_x = X[train_indices, :]
    train_y = Y[train_indices, :]

    return train_x, train_y, test_x, test_y