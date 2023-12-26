# 将转换到 np 格式的 torch 模型文件转换为 mindspore 支持读取的格式
from typing import (
    List, Tuple, Dict
)
import numpy as np
from mindspore import Tensor
import mindspore as ms

'''
# 参考的 np 文件转换
def _tensor_numpy(tensor:torch.Tensor):
    dt = tensor.dtype
    nt = 'float32'
    if dt == torch.float16:
        nt = 'float16'
    elif dt == torch.float32:
        nt = 'float32'
    elif dt == torch.float64:
        nt = 'float64'
    elif dt == torch.int64:
        nt = 'int64'
    val:np.ndarray = tensor.cpu().numpy()
    return val.astype(dtype=nt)

def change_model_into_ms(model_path:str, target_path:str):
    params = torch.load(model_path)
    np_dict = {}
    for key, val in params.items():
        print(f"handle--{key}")

        np_dict[key] = _tensor_numpy(val)
    
    np.save(target_path, np_dict)
'''

def generate_params_dict(total_layers,
                         mindspore_params_per_layer,
                         torch_params_per_layer,
                         mindspore_additional_params,
                         torch_additional_params):
    """
    Generate the total parameter mapping of mindspore and pytorch.

    Args:
        total_layers(int): The total layers of the net.
        mindspore_params_per_layer(list): The list of params per layer for the net of mindspore.
        torch_params_per_layer(list): The list of params per layer for the net of pytorch.
        mindspore_additional_params(list): The list of params outside the layer for the net of mindspore
        torch_additional_params(list): The list  of params outside the layer for the net of pytorch.

    Returns:
        A list of tuple. The first element is the parameter name of mindspore,
        the another is the parameter name of pytorch.
    """
    mapped_params = list(zip(mindspore_params_per_layer, torch_params_per_layer))
    ms_extend_param_list = []
    torch_extend_param_list = []
    for i in range(total_layers):
        for ms_para, torch_para in mapped_params:
            src = ms_para.format(i)
            tgt = torch_para.format(i)

            ms_extend_param_list.append(src)
            torch_extend_param_list.append(tgt)

    mapped_params = list(zip(mindspore_additional_params, torch_additional_params))
    for ms_para, torch_para in mapped_params:
        ms_extend_param_list.append(ms_para)
        torch_extend_param_list.append(torch_para)

    return list(zip(ms_extend_param_list, torch_extend_param_list))

def get_converted_ckpt(mapped_params:List[Tuple[str, str]], weight_dict:Dict[str, np.ndarray]):
    # to map
    np_ms_param_mapping= {}
    for ms_param, np_param in mapped_params:
        np_ms_param_mapping[np_param] = ms_param

    new_ckpt_list = []

    for np_param, value in weight_dict.items():
        if np_param not in np_ms_param_mapping:
            new_ckpt_list.append({
                "data" : Tensor(value),
                "name" : np_param
            })
            continue
        
        if 'output.dense.weight' in np_param or 'intermediate.dense.weight' in np_param:
            value = np.transpose(value, [1, 0])

        ms_param = np_ms_param_mapping[np_param]
        print( f"从np {np_param:<30} 映射到 ms {ms_param:<30}")

        new_ckpt_list.append({
            "data" : Tensor(value),
            "name" : ms_param
        })
    
    return new_ckpt_list


def main(model_path, out_path):
    # 参数映射，numpy Paramtes化，保存模型
    ms_name = [
        "bert_encoder.encoder.blocks.{}.attention.dense1.weight",
        "bert_encoder.encoder.blocks.{}.attention.dense1.bias",
        "bert_encoder.encoder.blocks.{}.attention.dense2.weight",
        "bert_encoder.encoder.blocks.{}.attention.dense2.bias",
        "bert_encoder.encoder.blocks.{}.attention.dense3.weight",
        "bert_encoder.encoder.blocks.{}.attention.dense3.bias",
        "bert_encoder.encoder.blocks.{}.attention.projection.weight",
        "bert_encoder.encoder.blocks.{}.attention.projection.bias",
        "bert_encoder.encoder.blocks.{}.layernorm2.gamma",
        "bert_encoder.encoder.blocks.{}.layernorm2.beta",
        "bert_encoder.encoder.blocks.{}.output.mapping.weight",
        "bert_encoder.encoder.blocks.{}.output.mapping.bias",
        "bert_encoder.encoder.blocks.{}.output.projection.weight",
        "bert_encoder.encoder.blocks.{}.output.projection.bias",
        "bert_encoder.encoder.blocks.{}.layernorm1.gamma",
        "bert_encoder.encoder.blocks.{}.layernorm1.beta",
    ]

    torch_name = [
        "encoder.layer.{}.attention.self.query.weight",
        "encoder.layer.{}.attention.self.query.bias",
        "encoder.layer.{}.attention.self.key.weight",
        "encoder.layer.{}.attention.self.key.bias",
        "encoder.layer.{}.attention.self.value.weight",
        "encoder.layer.{}.attention.self.value.bias",
        "encoder.layer.{}.attention.output.dense.weight",
        "encoder.layer.{}.attention.output.dense.bias",
        "encoder.layer.{}.attention.output.LayerNorm.weight",
        "encoder.layer.{}.attention.output.LayerNorm.bias",
        "encoder.layer.{}.intermediate.dense.weight",
        "encoder.layer.{}.intermediate.dense.bias",
        "encoder.layer.{}.output.dense.weight",
        "encoder.layer.{}.output.dense.bias",
        "encoder.layer.{}.output.LayerNorm.weight",
        "encoder.layer.{}.output.LayerNorm.bias",
    ]

    addition_mindspore = [
        "word_embedding.embedding_table",
        "embedding_postprocessor.full_position_embedding.embedding_table",
        "embedding_postprocessor.token_type_embedding.embedding_table",
        "embedding_postprocessor.layernorm.gamma",
        "embedding_postprocessor.layernorm.beta",
        "dense.weight",
        "dense.bias",
        "mlmloss.dense.weight",
        "mlmloss.dense.bias",
        "mlmloss.layernorm.gamma",
        "mlmloss.layernorm.beta",
        "mlmloss.vocab_dense.weight",
    ]

    addition_torch = [
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
        "pooler.dense.weight",
        "pooler.dense.bias",
        # "cls.predictions.transform.dense.weight",
        # "cls.predictions.transform.dense.bias",
        # "cls.predictions.transform.LayerNorm.gamma",
        # "cls.predictions.transform.LayerNorm.beta",
        # "cls.predictions.decoder.weight"

    ]

    ms_prefix = '_embedder.token_embedder_bert._bert.'
    np_prefix = '_embedder.token_embedder_bert._matched_embedder.transformer_model.'

    for idx, txt in enumerate(ms_name):
        ms_name[idx] = ms_prefix + txt

    for idx, txt in enumerate(addition_mindspore):
        addition_mindspore[idx] = ms_prefix + txt
    
    for idx, txt in enumerate(torch_name):
        torch_name[idx] = np_prefix + txt

    for idx, txt in enumerate(addition_torch):
        addition_torch[idx] = np_prefix + txt
    
    params_mapping_tuple_list = generate_params_dict(12, ms_name, torch_name, addition_mindspore, addition_torch)

    weight_dict = np.load(model_path, allow_pickle=True).item()
    new_ckpt = get_converted_ckpt(params_mapping_tuple_list, weight_dict)
    ms.save_checkpoint(new_ckpt, out_path)

if __name__ == '__main__':
    model_path = 'pretrain/np_best.npy'
    out_path = 'pretrain/ms_best.ckpt'
    main(model_path=model_path, out_path=out_path)