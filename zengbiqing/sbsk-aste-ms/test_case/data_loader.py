from span_model.dataset_readers.span_model import SpanModelReader
from pathlib import Path
from Hyperparameters import Hyperparameter as hp

def test_get_data():
    from span_model.dataset_readers.span_model import (
        SpanModelReader, get_span_model_reader
    )
    sr = get_span_model_reader()
    train_file_path = hp.__root__.joinpath('model_outputs/14lap_0/train.json')
    for data in sr.read(train_file_path):
        for key, val in SpanModelReader.extract_data(data).items():
            print(f"===={key}==>")
            print(val)
        # print(data)
        pass
    # sr._token_indexer.save_extend_vocab(hp.__root__ / 'output'/ 'temp')


def test_path():
    train_file_path = hp.__root__.joinpath('model_outputs/14lap/train.json').resolve()
    print(train_file_path.parent.name)

def file_type_test():
    from ms_allennlp_fix.field_type import ListField, SpanField, Token, TextField
    tf = TextField([Token('test'), Token('for'), Token('ll')])
    sp1 = SpanField(0, 1, tf)
    sp2 = SpanField(1, 2, tf)
    l1 = ListField([sp1, sp2])
    l2 = ListField([l1])
    print(l2)
    res = l2.as_tensor()
    print(type(res))
    print(res)