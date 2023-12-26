import mindspore
import torch
import numpy as np


def save_tensor_to_file(tensor, filename):
    # 将PyTorch张量转换为NumPy数组
    numpy_array = tensor.cpu().numpy()
    data_file = '/media/sqp/SQP_MAIN_DISK/100-代码/110-深度学习论文代码/Opengait-Mindspore/equal_data/data.npy'
    # 保存NumPy数组到文件
    np.save(data_file, numpy_array)


def load_tensor_from_file():
    # 从文件中加载NumPy数组
    data_file = '/media/sqp/SQP_MAIN_DISK/100-代码/110-深度学习论文代码/Opengait-Mindspore/equal_data/data.npy'
    loaded_array = np.load(data_file)

    return loaded_array  # 将NumPy数组转换为PyTorch张量


def load_ms_tensor_from_file():
    # 从文件中加载NumPy数组
    data_file = '/media/sqp/SQP_MAIN_DISK/100-代码/110-深度学习论文代码/Opengait-Mindspore/equal_data/data.npy'
    loaded_array = np.load(data_file)
    loaded_array = mindspore.Tensor(loaded_array)
    return loaded_array  # 将NumPy数组转换为PyTorch张量


def equal(data_numpy, decimal=6):
    loaded_numpy = load_tensor_from_file()
    # 比较原始张量和加载的张量
    print(np.testing.assert_almost_equal(data_numpy, loaded_numpy, decimal))


def run_model_and_save(test_model, input_data):
    output_data = test_model(input_data)
    save_tensor_to_file(output_data)
    return output_data


def run_model_and_equal(test_model, input_data, decimal=6):
    output_data = test_model(input_data)
    equal(output_data.numpy(), decimal)
    return output_data
