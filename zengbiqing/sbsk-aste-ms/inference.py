from Model import get_model
from Data import get_data_loader, build_input_path, TypeDataSet, TypeWork
import mindspore as ms
import numpy as np
from msadapter.pytorch import Parameter
from Hyperparameters import Hyperparameter as hp

def inference():
    from Model import get_model
    from Data import get_data_loader, build_input_path, TypeDataSet, TypeWork
    import mindspore as ms

    dl = get_data_loader()
    data_path = build_input_path(TypeDataSet._14lap, 0, TypeWork._test)
    dl._token_indexer.load_extend_vocab(hp.extend_vocab_path)
    data_generce = dl.read(data_path)

    span_model = get_model(dl._token_indexer)
    span_model.set_train(False)

    ms.load_checkpoint(
        str(hp.__root__/'pretrain'/'ms_best.ckpt'), 
        span_model
    )

    from fix.debug import show_dict_tensor_just_shape
    for idx, dd in enumerate(data_generce):
        span_model.set_text_input(
            dd['metadata'],
            dd['tree_info'],
        )
        ret = span_model(
            dd['text'],
            dd['spans'],
            dd['dep_span_children'],
            dd['ner_labels'],
            dd['relation_labels'],
            dd['dep_type_matrix'],
            dd['max_span_width'],
        )
        # show_dict_tensor_just_shape(ret)
        if idx == 30:
            break

    print(span_model.get_metrics())

if __name__ == '__main__':
    inference()