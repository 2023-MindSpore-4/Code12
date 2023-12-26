from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor
import torch
import re
import pickle
mlp_type = {}
replace_name_dict = {
    "normalize.bias": "normalize.beta",
    'normalize.weight': 'normalize.gamma',
    'bn1d.running_mean': 'bn1d.moving_mean',
    'bn1d.running_var': 'bn1d.moving_variance',
    'bn1d.weight':'bn1d.gamma',
     'bn1d.bias': "bn1d.beta",
    'gaitCorrectionModule.encoder.features.0.0.weight' : 'gaitCorrectionModule.encoder.features.0.features.0.weight',
    'gaitCorrectionModule.encoder.features.0.1.weight' : 'gaitCorrectionModule.encoder.features.0.features.1.gamma',
    'gaitCorrectionModule.encoder.features.0.1.bias'   : 'gaitCorrectionModule.encoder.features.0.features.1.beta',
    'gaitCorrectionModule.encoder.features.0.1.running_mean' : 'gaitCorrectionModule.encoder.features.0.features.1.moving_mean',
    'gaitCorrectionModule.encoder.features.0.1.running_var' : 'gaitCorrectionModule.encoder.features.0.features.1.moving_variance'

}
omit_weights_set={
    'smpl',
    'num_batches_tracked',
    'mobilenet'

}
def weight_name(name, is_normalize = False):
    if is_normalize:
        if name == 'weight':
            name = 'gamma'
        elif name == 'bias':
            name = 'beta'
        elif name == 'running_mean':
            name = 'moving_mean'
        elif name == 'running_var':
            name = 'moving_variance'
        return name
    else :
        return name
def mlp_weight_name(name, mlp_type):
    if mlp_type == 'BatchNorm1d':
        if name == 'weight':
            name = 'gamma'
        elif name == 'bias':
            name = 'beta'
        elif name == 'running_mean':
            name = 'moving_mean'
        elif name == 'running_var':
            name = 'moving_variance'
        return name
    elif mlp_type == 'mlp' :
        return name
    elif mlp_type == 'PReLU':
        return 'w'


def convert_name(pytorch_name, is_normalize, parameter):

    if 'gaitCorrectionModule.encoder' in pytorch_name:
        match = re.match(r'^gaitCorrectionModule\.encoder\.features\.(\d+)\.conv\.(\d+)\.(\d+)\.(\w+)$', pytorch_name)
        if match:
            idx1, idx2, inx3,param_name = match.groups()
            param_name = weight_name(param_name,is_normalize )
            output_line = f'gaitCorrectionModule.encoder.features.{idx1}.conv.{idx2}.features.{inx3}.{param_name}'
            return output_line
        match = re.match(r'^gaitCorrectionModule\.encoder\.features\.(\d+)\.conv\.(\d+)\.(\w+)$',pytorch_name)
        if match:
            idx1, idx2, param_name = match.groups()
            param_name = weight_name(param_name,is_normalize )
            output_line = f'gaitCorrectionModule.encoder.features.{idx1}.conv.{idx2}.{param_name}'
            return output_line
        match = re.match(r'^gaitCorrectionModule\.encoder\.features\.(\d+)\.(\d+)\.(\w+)$',pytorch_name)
        if match:
            idx1, idx2, param_name = match.groups()
            param_name = weight_name(param_name, is_normalize)
            output_line = f'gaitCorrectionModule.encoder.features.{idx1}.features.{idx2}.{param_name}'
            return output_line
    elif 'gaitCorrectionModule.mlp' in pytorch_name:
        match = re.match(r'^gaitCorrectionModule\.mlp\.(\d+).(\w+)$', pytorch_name)
        if match:
            idx1, param_name = match.groups()

            mlp_name =  f'gaitCorrectionModule.mlp.{idx1}'
            if param_name == 'weight' and is_normalize == False:
                mlp_type[mlp_name] = 'mlp'
            elif param_name == 'weight' and is_normalize == True and parameter.shape[0] == 1:
                mlp_type[mlp_name] = 'PReLU'
            elif param_name == 'weight' and is_normalize == True and parameter.shape[0] != 1:
                mlp_type[mlp_name] = 'BatchNorm1d'
            param_name = mlp_weight_name(param_name, mlp_type[mlp_name])
            output_line = f'gaitCorrectionModule.mlp.{idx1}.{param_name}'
            return output_line
    elif 'gaitCorrectionModule.out_mlp' in pytorch_name:
        match = re.match(r'^gaitCorrectionModule\.out_mlp\.(\d+).(\w+)$', pytorch_name)
        if match:
            idx1, param_name = match.groups()

            mlp_name = f'gaitCorrectionModule.out_mlp.{idx1}'
            if param_name == 'weight' and is_normalize == False:
                mlp_type[mlp_name] = 'mlp'
            elif param_name == 'weight' and is_normalize == True and parameter.shape[0] == 1:
                mlp_type[mlp_name] = 'PReLU'
            elif param_name == 'weight' and is_normalize == True and parameter.shape[0] != 1:
                mlp_type[mlp_name] = 'BatchNorm1d'
            param_name = mlp_weight_name(param_name, mlp_type[mlp_name])
            output_line = f'gaitCorrectionModule.out_mlp.{idx1}.{param_name}'
            return output_line
    for key, value in replace_name_dict.items():
        if pytorch_name.endswith(key):
            name = pytorch_name[:pytorch_name.rfind(key)]
            name = name + value
            return name
    return pytorch_name
def pytorch2mindspore(ckpt_name='res18_py.pth'):
    par_dict = torch.load(ckpt_name, map_location=torch.device('cpu'))
    # 从文件中读取列表
    with open('./ms_params_name_list.pkl', 'rb') as file:
        ms_params_name_list = pickle.load(file)
    new_params_list = []
    par_dict = par_dict['model']
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        is_continue = False
        for wight_name in omit_weights_set:
            if wight_name in name:
                is_continue = True
        if is_continue:
            continue
        print('========================py_name', name)

        name = convert_name(name, len(parameter.shape) == 1, parameter)



        print('========================ms_name', name)
        if name not in ms_params_name_list:
            print("ERROR")
            return
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, 'gait_checkpoint.ckpt')


if __name__ == '__main__':
    ckpt_name = "./experiment2-50000.pt"
    pytorch2mindspore(ckpt_name)
