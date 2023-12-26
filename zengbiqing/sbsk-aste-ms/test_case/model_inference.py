from pathlib import Path
from Hyperparameters import Hyperparameter as hp

def show_pertrain_info():
    import mindspore as ms
    p = Path(
        "/root/.mscache/mindspore/1.9/bertfinetune_squad_ascend_v190_squad_official_nlp_F1score86.87_exactmatch79.52.ckpt"
    ).resolve()
    if p.exists():
        state_dict = ms.load_checkpoint(str(p))
        for key, item in state_dict.items():
            print(key)
            print(item)
    else:
        print(f"no such file:{str(p)}")

def inference():
    from Model import get_model
    from Data import get_data_loader, build_input_path, TypeDataSet, TypeWork
    import mindspore as ms

    dl = get_data_loader()
    data_path = build_input_path(TypeDataSet._14lap, 0, TypeWork._train)
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

def show_model_info():
    from Model import get_model
    from Data import get_data_loader, build_input_path, TypeDataSet, TypeWork
    dl = get_data_loader()
    data_path = build_input_path(TypeDataSet._14lap, 0, TypeWork._train)
    g = dl.read(data_path)
    dd = next(iter(g))
    span_model = get_model(dl._token_indexer)
    for name, params in span_model.named_parameters():
        print(name)
        # print(params)

def show_torch_model_info():
    import msadapter.pytorch as torch
    from Hyperparameters import Hyperparameter as hp
    import mindspore as ms

def sample_train_test():
    '''一个简单的训练过程'''
    import mindspore as ms
    import msadapter.pytorch as torch
    import msadapter.pytorch.nn as nn
    import msadapter.pytorch.nn.functional as F
    from msadapter.pytorch.optim import AdamWeightDecay
    from random import random
    class Md(nn.Module):
        def __init__(self):
            super(Md, self).__init__()

            self.lin = nn.Linear(3, 3)
            self.weight = 1
        
        def forward(self, x):
            y = self.lin(x)
            loss = F.l1_loss(y, x) * self.weight
            return loss, y
        
        def set_weight(self, weight):
            self.weight = weight
    
    model = Md()

    optimizer = AdamWeightDecay(model.trainable_params())
    
    def gen(count = 100):
        for _ in range(count):
            data = {
                'x' : torch.ones(4, 3) * 100
            }
            yield data
    data = gen()
    
    def forward_func(x):
        model.set_weight(random())
        return model(**x)

    grad_func = ms.value_and_grad(forward_func, None, optimizer.parameters, has_aux=True)

    def train_step(x):
        (loss, _), grads = grad_func(x)
        optimizer(grads)
        return loss
    
    def train_loop():
        for idx, x in enumerate(data):
            loss = train_step(x)
            print(f"==={idx}==={loss}")
    
    train_loop()

def train_test(test_count = 10):
    from Model import get_model
    from Data import get_data_loader, build_input_path, TypeDataSet, TypeWork
    from msadapter.pytorch.optim import AdamWeightDecay
    from mindspore import value_and_grad

    dl = get_data_loader()
    dl._token_indexer.load_extend_vocab(hp.extend_vocab_path)
    data_path = build_input_path(TypeDataSet._14lap, 0, TypeWork._train)
    # datas = dl.read(data_path)
    # 测试一定值
    datas = []
    for idx, d in enumerate(dl.read(data_path)):
        datas.append(d)
        if idx == 1:
            break

    span_model = get_model(dl._token_indexer)

    optimizer = AdamWeightDecay(
        span_model.trainable_params(), weight_decay=0, learning_rate=0.001
    )

    # TODO 检查两种训练过程的区别，除一下 id 的错

    def post_fun( metadata, tree_info=None):
        span_model.set_text_input(
            metadata, tree_info
        )

    def forward_fn(
        text,
        spans,
        dep_span_children,
        ner_labels,
        relation_labels,
        dep_type_matrix,
        max_span_width,
    ):
        span_model_forward = span_model(
            text,
            spans,
            dep_span_children,
            ner_labels,
            relation_labels,
            dep_type_matrix,
            max_span_width,
        )
        loss = span_model_forward['loss']
        return loss, span_model_forward

    grad_fn = value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True
    )

    def train_step(data):
        post_fun(
            data['metadata'],
            data['tree_info'],
        )
        (loss, _), grads = grad_fn(
            data['text'],
            data['spans'],
            data['dep_span_children'],
            data['ner_labels'],
            data['relation_labels'],
            data['dep_type_matrix'],
            data['max_span_width'],
        )
        optimizer(grads)
        return loss
    
    def train_loop():
        for batch, data in enumerate(datas):
            loss = train_step(data)
            print(f"{batch}==={loss}")
            if batch == test_count -1:
                break

    train_loop()