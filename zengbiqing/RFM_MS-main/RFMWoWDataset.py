from data.Utils import *
import mindspore
import x2ms_adapter
import x2ms_adapter.nn_functional
import mindspore.nn
import numpy
import x2ms_adapter.nn as x2ms_nn




def get_selection_label(b, r, min_window_size=5, n_windows=4):
    # print(b.size())
    window_size = min_window_size
    bs = list()
    
    print('unfold之前b的维度')
    print(b.dim())
    print('unfold之前b的形状')
    print(b.shape)
    print(b)
    
    ### mindspore
    # c = x2ms_adapter.tensor_api.unsqueeze(b, 0)
    # d = x2ms_adapter.tensor_api.unsqueeze(c, 0).astype('float16')

    # unfoldop=ops.Custom(unfold)
    # k = unfold(d, 1, window_size, min_window_size)

    
    #
    # Unfold = mindspore.nn.Unfold(ksizes=[1,window_size,window_size,1], strides=[1,min_window_size,min_window_size,1], rates=[1,1,1,1])
    # print(Unfold(d).dim())
    # print(Unfold(d).shape)

    btuple = b.split(0,64)
    btensors = mindspore.ops.stack(btuple,0)
    bunfold = x2ms_adapter.tensor_api.unsqueeze(btensors, 0)
    print(bunfold.shape)

        # bs.append(x2ms_adapter.nn_functional.x2ms_pad(unfoldop(d,1, window_size, min_window_size), (0, min_window_size * n_windows - window_size)))
    # bs.append(x2ms_adapter.nn_functional.x2ms_pad(bunfold, (0, min_window_size * n_windows - window_size)))
    bs.append(bunfold)
    print(bs)
    b_segments = x2ms_adapter.cat(bs, dim=1)


    b_list = x2ms_adapter.tensor_api.tolist(b_segments)
    r_list = x2ms_adapter.tensor_api.tolist(r)

    overlap = [[len(set(seg).intersection(r_list[i])) for seg in b_list[i]] for i in range(len(b_list))]

    p_s = x2ms_adapter.tensor_api.detach(x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.x2ms_tensor(overlap)), dim=-1))
    return p_s


class RFMWoWDataset:
    def __init__(self, vocab2id, mode, samples, query, passage, min_window_size=5, num_windows=4, knowledge_len=300,
                 context_len=65, max_dec_length=80, n=1E10):
        super(RFMWoWDataset, self).__init__()

        self.min_window_size = min_window_size
        self.num_windows = num_windows
        self.knowledge_len = knowledge_len
        self.context_len = context_len
        self.max_dec_length = max_dec_length

        # WoW
        self.mode = mode
        self.samples = samples
        self.query = query
        self.passage = passage
        self.query_id = list()
        self.context_id = list()

        # 标量
        self.ids = list()
        self.contexts = list()
        self.queries = list()
        self.responses = list()
        self.unstructured_knowledges = list()
        self.dyn_vocab2ids = list()
        self.dyn_id2vocabs = list()
        self.example_id = list()

        # tensor
        self.id_arrays = list()
        self.context_arrays = list()
        self.query_arrays = list()
        self.response_arrays = list()
        self.dyn_response_arrays = list()
        self.unstructured_knowledge_arrays = list()

        self.ref_start_arrays = list()
        self.ref_end_arrays = list()

        self.dyn_map_arrays = list()
        self.vocab_map_arrays = list()
        self.vocab_overlap_arrays = list()

        self.selections = list()

        self.vocab2id = vocab2id
        self.n = n

        self.load()

    def load(self):
        for id in range(len(self.samples)):
            sample = self.samples[id]

            # 处理对话上下文
            context = self.query[sample['query_id']]
            context_ = [word.lower() for word in context]
            self.contexts.append(context_)
            if len(context_) > self.context_len:
                context_ = context_[-self.context_len:]
            elif len(context_) < self.context_len:
                context_ = context_ + [PAD_WORD] * (self.context_len - len(context_))
            self.context_arrays.append(x2ms_adapter.tensor_api.long(x2ms_adapter.x2ms_tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in context_],
                requires_grad=False)))

            # 处理背景知识
            # correct answer is always the first one
            temp_sample_knowledge_pool = sample['shifting_knowledge_pool'].copy()
            unstructured_knowledge_origin = []
            for pid in temp_sample_knowledge_pool:
                temp = self.passage[pid]
                for word in temp:
                    unstructured_knowledge_origin.append(word)
                if len(unstructured_knowledge_origin) > 256:
                    break
            unstructured_knowledge_origin = [word.lower() for word in unstructured_knowledge_origin]
            self.unstructured_knowledges.append(unstructured_knowledge_origin)
            unstructured_knowledge = unstructured_knowledge_origin
            if len(unstructured_knowledge) > self.knowledge_len:
                unstructured_knowledge = unstructured_knowledge[:self.knowledge_len]
            else:
                unstructured_knowledge = unstructured_knowledge + [PAD_WORD] * (self.knowledge_len - len(unstructured_knowledge))
            b = x2ms_adapter.tensor_api.long(x2ms_adapter.x2ms_tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in
                              unstructured_knowledge], requires_grad=False))
            if x2ms_adapter.tensor_api.x2ms_size(b)[0] > 256 or x2ms_adapter.tensor_api.x2ms_size(b)[0] == 0:
                print(len(unstructured_knowledge))
            self.unstructured_knowledge_arrays.append(b)

            # 处理ground-true背景知识
            bg_ref_start = -1
            bg_ref_end = -1
            if temp_sample_knowledge_pool[0] != 'K_0':
                bg_ref_start = 0
                bg_ref_end = len(self.passage[temp_sample_knowledge_pool[0]]) - 1
            self.ref_start_arrays.append(x2ms_adapter.x2ms_tensor([bg_ref_start], requires_grad=False))
            self.ref_end_arrays.append(x2ms_adapter.x2ms_tensor([bg_ref_end], requires_grad=False))

            # 处理回复
            response = sample['response']
            response = [word.lower() for word in response]
            self.responses.append([response])
            response = (response + [EOS_WORD])[:self.max_dec_length]
            r = x2ms_adapter.tensor_api.long(x2ms_adapter.x2ms_tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in response],
                requires_grad=False))
            self.response_arrays.append(r)

            self.selections.append(
                get_selection_label(b, x2ms_adapter.tensor_api.unsqueeze(r, 0), min_window_size=self.min_window_size,
                                    n_windows=self.num_windows))

            # from data.Utils import build_vocab
            dyn_vocab2id, dyn_id2vocab = build_vocab(unstructured_knowledge)  # 返回知识的vocab2id和id2vocab，特殊字符只有PAD
            self.dyn_vocab2ids.append(dyn_vocab2id)
            self.dyn_id2vocabs.append(dyn_id2vocab)

            self.dyn_response_arrays.append(
                x2ms_adapter.tensor_api.long(x2ms_adapter.x2ms_tensor([dyn_vocab2id.get(w) if w in dyn_vocab2id else 0 for w in response],
                             requires_grad=False)))
            self.dyn_map_arrays.append(
                x2ms_adapter.x2ms_tensor([dyn_vocab2id.get(w) for w in unstructured_knowledge], requires_grad=False))

            vocab_map = []
            vocab_overlap = []
            for i in range(len(dyn_id2vocab)):
                vocab_map.append(self.vocab2id.get(dyn_id2vocab[i], self.vocab2id[UNK_WORD]))  # 用背景知识用词表来表示
                if dyn_id2vocab[i] in self.vocab2id:
                    vocab_overlap.append(0.)
                else:
                    vocab_overlap.append(1.)
            self.vocab_map_arrays.append(x2ms_adapter.x2ms_tensor(vocab_map, requires_grad=False))  # 同上
            self.vocab_overlap_arrays.append(
                x2ms_adapter.x2ms_tensor(vocab_overlap, requires_grad=False))  # 如果背景知识词在词表存在为0，否则为1

            # e_id = sample['id']
            # self.example_id.append(e_id)

            self.ids.append(id)
            self.id_arrays.append(x2ms_adapter.tensor_api.long(x2ms_adapter.x2ms_tensor([id])))

            self.context_id.append(sample['context_id'])
            self.query_id.append(sample['query_id'])

            if len(self.contexts) >= self.n:
                break
        self.len = len(self.contexts)
        print('full data size: ', self.len)

    def __getitem__(self, index):
        id_arrays = self.id_arrays[index]
        contex_arrays = self.context_arrays[index]
        unstructured_knowledge_arrays = self.unstructured_knowledge_arrays[index]
        response_arrays = self.response_arrays[index]
        dyn_response_arrays = self.dyn_response_arrays[index]
        dyn_map_arrays = self.dyn_map_arrays[index]
        vocab_map_arrays = self.vocab_map_arrays[index]
        vocab_overlap_arrays = self.vocab_overlap_arrays[index]
        list_dvn_id2vocabs = numpy.array(list(self.dyn_id2vocabs[index].items()))
        list_dyn_vocab2ids = numpy.array(list(self.dyn_vocab2ids[index].items()))
        ids_dyn_id2vocabs = (self.ids[index], list_dvn_id2vocabs)
        ids_dyn_vocab2ids = (self.ids[index], list_dyn_vocab2ids)
        selections = self.selections[index]
        ref_start_arrays = self.ref_start_arrays[index]
        ref_end_arrays = self.ref_end_arrays[index]
        return id_arrays, contex_arrays, unstructured_knowledge_arrays, response_arrays, dyn_response_arrays, dyn_map_arrays, vocab_map_arrays, vocab_overlap_arrays, ids_dyn_id2vocabs, ids_dyn_vocab2ids, selections, ref_start_arrays, ref_end_arrays

    def __len__(self):
        return self.len

    def input(self, id):
        return self.contexts[id]

    def output(self, id):
        return self.responses[id]

    def background(self, id):
        return self.unstructured_knowledges[id]
    
    def c_id(self, id):
        return self.context_id[id]
    
    def q_id(self, id):
        return self.query_id[id]


# def pad_sequence(sequences,padding_value=0):
#     max = mindspore.numpy.size(sequences[0],-1)
#     for seq in sequences:
#         if mindspore.numpy.size(seq,-1) > max:
#             max=mindspore.numpy.size(seq,-1)
#     for seq in sequences:
#         seq = mindspore.ops.padding(seq,max)
#     return mindspore.ops.stack(sequences,0)

def pad_sequence(sequences, batch_first=False, padding_value=0.0):

    sequences = list(sequences)

    tensor_value = sequences[0].asnumpy()
    turn_sequences = mindspore.Tensor([tensor_value])

    return turn_sequences
# def pad_sequence(sequences, batch_first=False, padding_value=0.0):
#     # sequences = mindspore.ops.tuple_to_array(sequences)
#     # sequences = mindspore.ops.split(sequences,0,sequences.shape[0])
#     max_length = max([seq.shape[0] for seq in sequences])
#     padded_sequences = []
#     for seq in sequences:
#         length = seq.shape[0]
#         pad_size = max_length - length
#         pad = mindspore.ops.Pad(((0, pad_size),))
#         padded_seq = pad(seq)
#         padded_sequences.append(padded_seq)
#     padded_sequences = mindspore.ops.stack(padded_sequences)
#     print(padded_sequences)
#     return padded_sequences
# def pad_sequence(sequences, batch_first=False, padding_value=0.0):
#     max_length = max([seq.shape[0] for seq in sequences])
#     trailing_dims = sequences[0].shape[1:]
#     out_dims = (len(sequences), max_length) + trailing_dims if batch_first else (max_length, len(sequences)) + trailing_dims
#     out_tensor = mindspore.Tensor(numpy.full(out_dims, padding_value))
#     for i, tensor in enumerate(sequences):
#         length = tensor.shape[0]
#         # use index to slice the tensor along the first dimension
#         if batch_first:
#             out_tensor[i, :length, ...] = tensor
#         else:
#             out_tensor[:length, i, ...] = tensor
#     return out_tensor


def collate_fn(data):
    print(data)
    print(zip(*data))
    id_a, context_a, unstructured_knowledge_a, response_a, dyn_response_a, dyn_map, vocab_map, vocab_overlap, dyn_id2vocab, dyn_vocab2id, selection, ref_start, ref_end = zip(
        *data)
    print("is it collate_fn accept the data successfully?")

    return {'id': x2ms_adapter.cat(id_a),
            'context': pad_sequence(context_a, batch_first=True),
            'response': pad_sequence(response_a, batch_first=True),
            'unstructured_knowledge': pad_sequence(unstructured_knowledge_a, batch_first=True),
            'dyn_response': pad_sequence(dyn_response_a, batch_first=True),
            'dyn_map': pad_sequence(dyn_map, batch_first=True),
            'vocab_map': pad_sequence(vocab_map, batch_first=True),
            'vocab_overlap': pad_sequence(vocab_overlap, batch_first=True, padding_value=1.),
            # 'dyn_id2vocab': dyn_id2vocab,
            # 'dyn_vocab2id': dyn_vocab2id,
            'selection': x2ms_adapter.cat(selection),
            'ref_start': x2ms_adapter.cat(ref_start),
            'ref_end': x2ms_adapter.cat(ref_end)}
