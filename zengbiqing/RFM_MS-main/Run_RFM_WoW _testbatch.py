import os
os.system("pip install dask[complete]")
os.system("pip install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install bcolz")
os.system("pip install regex")
from RFMWoWDataset import *
from RFM import *
from mindspore import Tensor
from trainers.TestTrainer import  *
import argparse
from data.Utils_WoW import *
from x2ms_adapter.optimizers import optim_register
import mindspore
import x2ms_adapter
import x2ms_adapter.distributed as x2ms_distributed
import x2ms_adapter.datasets
import mindspore.dataset as ds
from mindspore import context
import nltk
nltk.data.path.append("/home/ma-user/modelarts/user-job-dir/RFM-main_x2ms/nltk_data")
context.set_context(mode=context.PYNATIVE_MODE)
import numpy as np

def train(args):
    data_path = args.data_path

    if x2ms_adapter.is_cuda_available():
        x2ms_distributed.init_process_group(backend='NCCL', init_method='env://')

    # cudnn.enabled = True
    # print(mindspore.__version__)
    # print(torch.version.cuda)
    # print(cudnn.version())

    init_seed(123456)

    batch_size = 32

    output_path = args.model_path

    vocab2id, id2vocab, id2freq = load_vocab(data_path + 'wow_input_output.vocab', t=args.min_vocab_freq)

    if not os.path.exists(data_path + 'glove.6B.300d.txt' + '.dat'):
        prepare_embeddings(data_path + 'glove.6B.300d.txt')
    emb_matrix = load_embeddings(data_path + 'glove.6B.300d.txt', id2vocab, args.embedding_size)

    samples, query, passage = load_default(args.dataset, data_path + args.dataset + '.answer',
                                           data_path + args.dataset + '.passage',
                                           data_path + args.dataset + '.pool',
                                           data_path + args.dataset + '.qrel',
                                           data_path + args.dataset + '.query')

    train_samples, dev_samples, test_seen_samples, test_unseen_samples = split_data(args.dataset,
                                                                                    data_path + args.dataset + '.split',
                                                                                    samples)
    print("The number of train_samples:", len(train_samples))

    # train_dataset = RFMWoWDataset(vocab2id, args.mode, train_samples, query, passage, args.min_window_size,
    #                               args.num_windows, args.knowledge_len, args.context_len)
    # print(train_dataset[0])
    #
    # new_dataset = ds.GeneratorDataset(train_dataset,column_names=["id_arrays", "context_arrays", "unstructured_knowledge_arrays", "response_arrays", "dyn_response_arrays", "dyn_map_arrays", "vocab_map_arrays", "vocab_overlap_arrays", "ids_dyn_id2vocabs", "ids_dyn_vocab2ids", "selections", "ref_start_arrays", "ref_end_arrays"])

    # for data in new_dataset.create_dict_iterator():
    #     print(data['id_arrays'])

    id = Tensor([[11475]])
    context = Tensor([[9182,    99,  5250, 53564,  9182,    31, 37045,    12,   450,  1496,  9978,    43,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]])
    # context = Tensor(np.random.randint(low=1, high=100000, size=(1,65)))
    response = Tensor([[9182,     2,    45,   180,  7388,    20,   606,  5996,    12,    45,  9206,     9,  8200,    16,    22,    25,  7388,    20,  9182,  3866,  4143,   776,    12,  4845,
    184,    43,  9182,     2,  9182,    17,    18,    23,  9197,    22,    59,  3910,  2026,  7847,    20,    54,  9198,  9199,    57,  8592,    22,  9200,    31,  6329,
    43,  9182,     2,    58,    17,   750,    71,    45,   131,    20,    18,  7057,    22,  4480,    22,    14,    71,    18,  9201,    22,    14,   425,    29,    18,
    9202,  9203,    71,    49,  4548,    43,  9182,     2,  9199,    51,    52,  3800,    97,   258,  5153,    91,   137,  4144,   249,  9204,    71,  9205,    43,     2,
     2,     2,  9182,     2,    71,   113,    22,    45,  4387,    20,  9207,  1007,   750,  9182,  3866,    22,   757,    45,  5217,    31,  9208,    22,    88,   750,
    58,   399,    18,  3940,   223,    29,    54,  9209,    57,    22,    18,  9210,  4491,   683,    54,  3908,  2015,    57,    43,  9182,     2,    45,  8592,    20,
    45,  9199,  5303,    68,   215,  4236,  3908,  1341,    31,  1741,   305,  3901,    12,  3545,    45,  4004,    43,  9182,     2,   840,  3833,    22,    45,  9267,
    124,  9320,    22, 32177,    22,    31,  9200,    43,  9182,     2,    45,  6064,    17, 11756,    12,  2422,  9199, 34817,    22,   269,   124,  1666,  6329,    12,
    9187,  5325,    22, 34818,  9182,    71,  4101,   131,    43,  9182,     2,   979,    45,  9187,  5325,    17, 34819,    97, 12345,    22,    58,    17,    53,  9182,
    4593,    43,  9182,     2,    45,  4593,   549,   402,   305, 27048,    31,  2351,   399,   144,   166,  4455,    96,  9187,  4632,    31,  9187,  9248,    43,  9182,
    2,  3797,  9182,    22,   549,    53,  3908,  9182,    22,  1141,  9187,  4632,    31,  9187,  9248,    71]])
    # unstructured_knowledge = Tensor([[88,   17,  967, 2707, 1906, 9182, 4585,  776,   12, 4845,  184,   43, 1416,  124,  117, 1418,   22, 4048, 3574, 1906,    3]])
    unstructured_knowledge = Tensor(np.random.randint(low=1, high=10000, size=(1,256)))
    dyn_response = Tensor([[72, 22,  0,  0,  0,  1,  0, 18,  9, 19, 20, 21,  0, 95,  0,  0, 14,  0,  0,  0,  0]])
    # dyn_response = Tensor(np.random.randint(low=1, high=100, size=(1, 21)))
    dyn_map = Tensor([[1,   2,   3,   4,   5,   6,   7,   8,   9,   3,  10,  11,  12,  13,  14,  15,   5,   6,   1,  16,  17,  18,   9,  19,
    20,  21,   1,   2,   1,  22,  23,  24,  25,  14,  26,  27,  28,  29,   6,  30,  31,  32,  33,  34,  14,  35,  36,  37,
    21,   1,   2,  38,  22,  39,  40,   3,  41,   6,  23,  42,  14,  43,  14,  44,  40,  23,  45,  14,  44,  46,  47,  23,
    48,  49,  40,  50,  51,  21,   1,   2,  32,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  40,  63,  21,  64,
    2,  64,   1,   2,  40,  65,  14,   3,  66,   6,  67,  68,  39,   1,  16,  14,  69,   3,  70,  36,  71,  14,  72,  39,
    38,  73,  23,  74,  75,  47,  30,  76,  33,  14,  23,  77,  78,  79,  30,  80,  81,  33,  21,   1,   2,   3,  34,   6,
    3,  32,  82,  83,  84,  85,  80,  86,  36,  87,  88,  89,   9,  90,   3,  91,  21,   1,   2,  92,  93,  14,   3,  94,
    95,  96,  14,  97,  14,  36,  35,  21,   1,   2,   3,  98,  22,  99,   9, 100,  32, 101,  14, 102,  95, 103,  37,   9,
    104, 105,  14, 106,   1,  40, 107,  41,  21,   1,   2, 108,   3, 104, 105,  22, 109,  55, 110,  14,  38,  22, 111,   1,
    112,  21,   1,   2,   3, 112, 113, 114,  88, 115,  36, 116,  73, 117, 118, 119, 120, 104, 121,  36, 104, 122,  21,   1,
    2, 123,   1,  14, 113, 111,  80,   1,  14, 124, 104, 121,  36, 104, 122,  40]])
    vocab_map = Tensor([[0,  9182,     2,    45,   180,  7388,    20,   606,  5996,    12,  9206,     9,  8200,    16,    22,    25,  3866,  4143,   776,  4845,   184,    43,    17,    18,
    23,  9197,    59,  3910,  2026,  7847,    54,  9198,  9199,    57,  8592,  9200,    31,  6329,    58,   750,    71,   131,  7057,  4480,    14,  9201,   425,    29,
    9202,  9203,    49,  4548,    51,    52,  3800,    97,   258,  5153,    91,   137,  4144,   249,  9204,  9205,     2,   113,  4387,  9207,  1007,   757,  5217,  9208,
    88,   399,  3940,   223,  9209,  9210,  4491,   683,  3908,  2015,  5303,    68,   215,  4236,  1341,  1741,   305,  3901,  3545,  4004,   840,  3833,  9267,   124,
    9320, 32177,  6064, 11756,  2422, 34817,   269,  1666,  9187,  5325, 34818,  4101,   979, 34819, 12345,    53,  4593,   549,   402, 27048,  2351,   144,   166,  4455,
    96,  4632,  9248,  3797,  1141]])
    vocab_overlap = Tensor([[0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]])
    selection = Tensor([[ 1.11443149e-02,  4.09976440e-03,  1.11443149e-02,  1.11443149e-02,  1.11443149e-02,  8.23459700e-02,  8.23459700e-02,  3.02933883e-02,  1.11443149e-02,  4.09976440e-03,  4.09976440e-03,  1.11443149e-02,
    3.02933883e-02,  1.11443149e-02,  4.09976440e-03,  1.11443149e-02,  1.11443149e-02,  4.09976440e-03,  4.09976440e-03,  3.02933883e-02,  4.09976440e-03,  4.09976440e-03,  4.09976440e-03,  1.11443149e-02,
    1.11443149e-02,  1.11443149e-02,  4.09976440e-03,  3.02933883e-02,  4.09976440e-03,  3.02933883e-02,  4.09976440e-03,  4.09976440e-03,  1.11443149e-02,  4.09976440e-03,  3.02933883e-02,  4.09976440e-03,
    4.09976440e-03,  4.09976440e-03,  4.09976440e-03,  1.11443149e-02,  3.02933883e-02,  1.11443149e-02,  3.02933883e-02,  3.02933883e-02,  1.11443149e-02,  3.02933883e-02,  1.11443149e-02,  3.02933883e-02,
    1.11443149e-02,  1.11443149e-02,  3.02933883e-02,  1.11443149e-02,  1.11443149e-02,  3.02933883e-02,  3.02933883e-02,  4.09976440e-03,  4.09976440e-03,  4.09976440e-03,  4.09976440e-03,  3.02933883e-02,
    3.02933883e-02,  1.11443149e-02,  1.11443149e-02,  4.09976440e-03]])
    ref_start = Tensor([[0]])
    ref_end = Tensor([[25]])

    new_dataset = {
            'id': id,
            'context': context,
            'response': response,
            'unstructured_knowledge': unstructured_knowledge,
            'dyn_response': dyn_response,
            'dyn_map': dyn_map,
            'vocab_map': vocab_map,
            'vocab_overlap': vocab_overlap,
            # 'dyn_id2vocab': dyn_id2vocab,
            # 'dyn_vocab2id': dyn_vocab2id,
            'selection': selection,
            'ref_start': ref_start,
            'ref_end': ref_end}

    model = RFM(args.min_window_size, args.num_windows, args.embedding_size, args.knowledge_len, args.context_len,
                args.hidden_size, vocab2id, id2vocab, max_dec_len=70,
                beam_width=1, emb_matrix=emb_matrix)
    init_params(model, escape='embedding')

    # model_optimizer = nn.Adam(model.trainable_params(),  learning_rate=0.0001)
    model_optimizer = optim_register.adam(x2ms_adapter.parameters(model), lr=0.0001)
    trainer = DefaultTrainer(model, args.local_rank)

    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in x2ms_adapter.parameters(model):
        mulValue = x2ms_adapter.tensor_api.prod(np, x2ms_adapter.tensor_api.x2ms_size(param))
        Total_params += mulValue  # total parameters
        if param.requires_grad:
            Trainable_params += mulValue  # trainable parameters
        else:
            NonTrainable_params += mulValue  # non-trainable parameters

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

    # for i in range(10):
    #     trainer.train_epoch('ds_train', train_dataset, collate_fn, batch_size, i, model_optimizer)

    for i in range(1):
        if i == 0:
            train_embedding(model)
        trainer.train_epoch('fb_mle_mcc_ds_train', new_dataset, collate_fn, batch_size, i, model_optimizer)
        # multi_schedule.step()
        # trainer.serialize(i, output_path=output_path)


# def test(args):
#     data_path = 'dataset/wizard_of_wikipedia/'
#
#     # cudnn.enabled = True
#     # print(mindspore.__version__)
#     # print(torch.version.cuda)
#     # print(cudnn.version())
#
#     init_seed(123456)
#
#     batch_size = 32
#
#     output_path = 'model/' + 'wizard_of_wikipedia/'
#
#     vocab2id, id2vocab, id2freq = load_vocab(data_path + 'wow_input_output.vocab', t=args.min_vocab_freq)
#
#     samples, query, passage = load_default(args.dataset, data_path + args.dataset + '.answer',
#                                            data_path + args.dataset + '.passage',
#                                            data_path + args.dataset + '.pool',
#                                            data_path + args.dataset + '.qrel',
#                                            data_path + args.dataset + '.query')
#
#     train_samples, dev_samples, test_seen_samples, test_unseen_samples = split_data(args.dataset,
#                                                                                     data_path + args.dataset + '.split',
#                                                                                     samples)
#     print("The number of test_seen_samples:", len(test_seen_samples))
#     print("The number of test_unseen_samples:", len(test_unseen_samples))
#
#     test_seen_dataset = RFMWoWDataset(vocab2id, args.mode, test_seen_samples, query, passage, args.min_window_size,
#                                       args.num_windows, args.knowledge_len, args.context_len)
#
#     test_unseen_dataset = RFMWoWDataset(vocab2id, args.mode, test_unseen_samples, query, passage, args.min_window_size,
#                                         args.num_windows, args.knowledge_len, args.context_len)
#
#     for i in range(30):
#         print('epoch ' + str(i))
#         file = output_path + 'model/' + str(i) + '.pkl'
#
#         if os.path.exists(file):
#             model = RFM(args.min_window_size, args.num_windows, args.embedding_size, args.knowledge_len,
#                         args.context_len, args.hidden_size, vocab2id, id2vocab, max_dec_len=70, beam_width=1)
#             x2ms_adapter.load_state_dict(model, x2ms_adapter.load(file))
#             trainer = DefaultTrainer(model, None)
#             # trainer.test('test', dev_dataset, collate_fn, batch_size, i, output_path=output_path)
#             # seen
#             print('test_seen:')
#             trainer.test('test', test_seen_dataset, collate_fn, batch_size, 100 + i, output_path=output_path,
#                          test_type=args.test)
#             # unseen
#             print('test_unseen:')
#             trainer.test('test', test_unseen_dataset, collate_fn, batch_size, 1000 + i, output_path=output_path,
#                          test_type=args.test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--test", type=str, default='SR')
    parser.add_argument("--dataset", type=str, default='wizard_of_wikipedia')
    parser.add_argument("--version", type=str, default='oracle')  # background version
    parser.add_argument("--embedding_size", type=int, default=300)  # embedding size
    parser.add_argument("--hidden_size", type=int, default=256)  # hidden size
    parser.add_argument("--min_window_size", type=int, default=4)  # the minimum size of slide window
    parser.add_argument("--num_windows", type=int, default=1)  # the stride of slide window
    parser.add_argument("--knowledge_len", type=int, default=256)  # background knowledge length
    parser.add_argument("--context_len", type=int, default=65)  # context length
    parser.add_argument("--min_vocab_freq", type=int, default=10)  # the minimum size of word frequency
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    # if args.mode == 'test':
    #     test(args)
    if args.mode == 'train':
        train(args)
