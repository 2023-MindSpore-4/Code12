import re

import mindspore
import torch
from mindspore import Tensor

from demo import get_parser_infer, parse_args
from mindyolo import create_model

bn_pt2ms = {
    "running_mean": "moving_mean",
    "running_var": "moving_variance",
    "weight": "gamma",
    "bias": "beta"
}


def find_bn(param_name, param_name1):
    if param_name == 'bn':
        return bn_pt2ms[param_name1]
    else:
        return param_name1


def get_idx(idx1, idx2):
    idx1 = int(idx1) - 2
    return idx1 * 2 + int(idx2) + 1


def convert_name(pytorch_name):
    if 'backbone.backbone.stem.' in pytorch_name:
        match = re.match(r'^backbone\.backbone\.stem\.conv\.(\w+)\.(\w+)$', pytorch_name)
        if match:
            param_name, param_name1 = match.groups()
            param_name1 = find_bn(param_name, param_name1)
            return f'model.model.0.conv.{param_name}.{param_name1}'
    elif 'backbone.backbone.dark' in pytorch_name:
        match = re.match(r'^backbone\.backbone\.dark(\d+)\.(\d+)\.(\w+)\.(\w+)$', pytorch_name)
        if match:
            idx1, idx2, param_name, param_name1 = match.groups()
            param_name1 = find_bn(param_name, param_name1)
            return f'model.model.{get_idx(idx1, idx2)}.{param_name}.{param_name1}'
        match = re.match(r'^backbone\.backbone\.dark(\d+)\.(\d+)\.(\w+)\.(\w+)\.(\w+)$', pytorch_name)
        if match:
            idx1, idx2, param_name0, param_name, param_name1 = match.groups()
            param_name1 = find_bn(param_name, param_name1)
            return f'model.model.{get_idx(idx1, idx2)}.{param_name0}.{param_name}.{param_name1}'
        match = re.match(r'^backbone\.backbone\.dark(\d+)\.(\d+).m\.(\d+)\.(\w+)\.(\w+)\.(\w+)$', pytorch_name)
        if match:
            idx1, idx2, idx3, param_name0, param_name, param_name1 = match.groups()
            param_name1 = find_bn(param_name, param_name1)
            return f'model.model.{get_idx(idx1, idx2)}.m.{idx3}.{param_name0}.{param_name}.{param_name1}'
    pytorch_name = pytorch_name.replace('backbone.lateral_conv0', 'model.model.10')
    pytorch_name = pytorch_name.replace('backbone.C3_p4', 'model.model.13')
    pytorch_name = pytorch_name.replace('backbone.reduce_conv1', 'model.model.14')
    pytorch_name = pytorch_name.replace('backbone.C3_p3', 'model.model.17')
    pytorch_name = pytorch_name.replace('backbone.bu_conv2', 'model.model.18')
    pytorch_name = pytorch_name.replace('backbone.C3_n3', 'model.model.20')
    pytorch_name = pytorch_name.replace('backbone.bu_conv1', 'model.model.21')
    pytorch_name = pytorch_name.replace('backbone.C3_n4', 'model.model.23')
    pytorch_name = pytorch_name.replace('head', 'model.model.24')

    if '.bn.' in pytorch_name:
        pytorch_name = pytorch_name.split('.')
        pytorch_name[-1] = bn_pt2ms[pytorch_name[-1]]
        pytorch_name = '.'.join(pytorch_name)
    return pytorch_name


def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')['model']
    pt_params = {}

    for last_name in par_dict:
        parameter = par_dict[last_name]
        if len(parameter.shape) != 0:
            new_name = convert_name(last_name)
            if new_name in pt_params:
                print(last_name, new_name)

                print("ERROR")
                return
            pt_params[new_name] = parameter.numpy()
    return pt_params


pth_path = "bytetrack_x_mot17.pth.tar"
pt_param = pytorch_params(pth_path)
parser = get_parser_infer()
args = parse_args(parser)
network = create_model(
    model_name=args.network.model_name,
    model_cfg=args.network,
    num_classes=args.data.nc,
    sync_bn=False,
    checkpoint_path='yolox-x.ckpt',
)


def mindspore_params(model):
    ms_params = {}
    for param in model.get_parameters():
        ms_name = param.name
        value = param.data.asnumpy()
        ms_params[ms_name] = value
    return ms_params


ms_param = mindspore_params(network)
print('=' * 20)
shapes = {}
pp = []
for name in pt_param:
    shapes[name] = pt_param[name].shape
al = 0
for name in ms_param:
    if name in shapes and ms_param[name].shape == shapes[name]:
        del shapes[name]
    else:
        al += 1

        # print(name, ms_param[name].shape)
for name in shapes:
    al += 1
    print(name, 0, shapes[name])
print(al)
new_params_list = []

for name in ms_param:
    ms_value = pt_param[name]
    new_params_list.append({"name": name, "data": Tensor(ms_value)})
mindspore.save_checkpoint(new_params_list, 'yolox-x.ckpt')
