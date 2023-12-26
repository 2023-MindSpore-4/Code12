from span_model.dataset_readers.span_model import get_span_model_reader
from Hyperparameters import Hyperparameter as hp

class TypeDataSet:
    _14lap = '14lap'
    _14res = '14res'
    _15res = '15res'
    _16res = '16res'

    _all = [_14lap, _14res, _15res, _16res]

class TypeWork:
    _dev = 'dev'
    _test = 'test'
    _train = 'train'

    _all = [_dev, _test, _train]

def build_input_path(data_set, seed, work):

    data_path = hp.data_path / f'{data_set}_{seed}' / f'{work}.json'

    return data_path


def get_data_loader():
    smr = get_span_model_reader()
    return smr