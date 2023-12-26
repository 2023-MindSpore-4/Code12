import json, argparse
from pathlib import Path
import mindspore as ms

class Hyperparameter:
    # path
    __root__ = Path(__file__).absolute().parent
    work_root = __root__/"output"
    work_space = work_root/"temp"
    data_path = __root__ / 'data_inputs'
    vocabulary_path = __root__ / 'pretrain' / 'vocab.txt'
    extend_vocab_path = __root__ / 'pretrain' /'extend_vocab_path'

    # # train kwargs
    # train_kwargs = {
    #     "names": "14lap,14res,15res,16res",
    #     "seeds": "0,0,0,0",
    #     "trainer__cuda_device": "0",
    #     "trainer__num_epochs": 10,
    #     "trainer__checkpointer__num_serialized_models_to_keep": 1,
    #     "model__span_extractor_type": "endpoint",
    #     "model__modules__relation__use_single_pool": False,
    #     "model__relation_head_type": "proper",
    #     "model__use_span_width_embeds": True,
    #     "model__modules__relation__use_distance_embeds": True,
    #     "model__modules__relation__use_pair_feature_multiply": False,
    #     "model__modules__relation__use_pair_feature_maxpool": False,
    #     "model__modules__relation__use_pair_feature_cls": False,
    #     "model__modules__relation__use_span_pair_aux_task": False,
    #     "model__modules__relation__use_span_loss_for_pruners": False,
    #     "model__loss_weights__ner": 1.0,
    #     "model__modules__relation__spans_per_word": 0.5,
    #     "model__modules__relation__neg_class_weight": -1,
    # }

    # pretrain_config
    pretrain_token_embedd_name = 'bert_base_uncased'
    pretrain_token_embedd_seq_len = 512


    params = json.load(open("./training_config/config.json"))
    # trainer
    num_epoch = 10
    validation_metric = 'MEAN__relation_f1'

    # mindspore
    mode = ms.PYNATIVE_MODE
    device = 'GPU'

    # dataloader
    use_generater = True
    batch_size = 10
    
hp = Hyperparameter

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", help="setting the workspace, saving the info during run", default=None)
parser.add_argument("--pretrain_dir", help="location of pretrain vocab", default=None)
parser.add_argument("--device", help="compute space", default=hp.device)
parser.add_argument("--use_generater", help="if False will full load data at first", default=hp.use_generater)
parser.add_argument("--batch_size", help="the batchsize trainer using", default=hp.batch_size, type=int)
parser.add_argument("--grap_model", help="使用什么模式运行模型", action="store_true")

args = parser.parse_args()
if args.work_dir is not None:
    work_dir = args.work_dir
    if work_dir[0] == '/':
        hp.work_space = Path(work_dir)
    else:
        hp.work_space = hp.work_root / f"{work_dir}"
    hp.work_space.mkdir(exist_ok=True, parents=True)

if args.pretrain_dir is not None:
    pretrain_dir = Path(args.pretrain_dir)
    hp.vocabulary_path = pretrain_dir / 'vocab.txt'
    hp.extend_vocab_path = pretrain_dir / 'extend_vocab_path'

hp.device = args.device
hp.use_generater = args.use_generater
hp.batch_size = args.batch_size

if args.grap_model:
    hp.mode = ms.GRAPH_MODE