import mindspore as ms
import mindspore.nn as nn
def show_model_info(model:nn.Cell):
    for name, paramter in model.parameters_and_names():
        print(name)
        print(paramter)