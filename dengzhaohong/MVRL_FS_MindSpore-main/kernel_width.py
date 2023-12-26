import numpy as np

def kernel_width(C, data, min_kernel, max_kernel):
    kernel_width = np.zeros((C.shape[0], data.shape[1]))
    n_samples = data.shape[0]
    for j in range(data.shape[1]):
        for k in range(C.shape[0]):
            kernel_width[k, j] = np.linalg.norm(data[:, j] - np.ones((n_samples, 1)) * C[k, j])
    
    kernel_width_sum = np.tile(np.sum(kernel_width, axis=0), (C.shape[0], 1))
    kernel_width = kernel_width / kernel_width_sum
    
    kernel_width = adjust_scale(kernel_width, min_kernel, max_kernel)
    
    return kernel_width

def membership_matrix(C, data):
    norm_matrix = np.zeros((C.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        data_vector = data[i, :]
        for k in range(C.shape[0]):
            norm_matrix[k, i] = np.linalg.norm(data_vector - C[k, :])
    
    norm_sum = np.tile(np.sum(norm_matrix, axis=0), (C.shape[0], 1))
    U = norm_matrix / norm_sum
    return U

def adjust_scale(inputMatrix, minVal, maxVal):
    minmum = np.min(inputMatrix)
    maxmum = np.max(inputMatrix)
    outputMatrix = (inputMatrix - minmum) / (maxmum - minmum)
    outputMatrix = minVal + outputMatrix * (maxVal - minVal)
    return outputMatrix