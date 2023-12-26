import mindspore


def adjust_scale(input_matrix, min_val, max_val):
    minimum = mindspore.ops.min(input_matrix)
    maximum = mindspore.ops.max(input_matrix)
    output_matrix = (input_matrix - minimum) / (maximum - minimum)
    output_matrix = min_val + output_matrix * (max_val - min_val)
    
    return output_matrix

def membership_matrix(C, data):
    norm_matrix = mindspore.ops.zeros((C.size(0), data.size(0)))
    
    for i in range(data.size(0)):
        data_vector = data[i, :]
        for k in range(C.size(0)):
            norm_matrix[k, i] = mindspore.ops.norm(data_vector - C[k, :])
    
    norm_sum = mindspore.ops.sum(norm_matrix, dim=1).unsqueeze(1)
    U = norm_matrix / norm_sum
    
    return U

def kernel_width(C, data, min_kernel, max_kernel):
    kernel_width = mindspore.ops.zeros((C.size(0), data.size(1)))
    n_samples = data.size(0)
    
    for j in range(data.size(1)):
        for k in range(C.size(0)):
            kernel_width[k, j] = mindspore.ops.norm(data[:, j] - mindspore.ops.ones(n_samples, 1) * C[k, j])
    
    kernel_width_sum = mindspore.ops.sum(kernel_width, axis=0).unsqueeze(0)
    kernel_width = kernel_width / kernel_width_sum
    
    kernel_width = adjust_scale(kernel_width, min_kernel, max_kernel)
    
    return kernel_width