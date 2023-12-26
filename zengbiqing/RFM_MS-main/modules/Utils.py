import numpy as np
import pickle
import bcolz
import mindspore
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = x2ms_adapter.tensor_api.x2ms_size(box_a, 0)
    B = x2ms_adapter.tensor_api.x2ms_size(box_b, 0)
    max_xy = x2ms_adapter.x2ms_min(x2ms_adapter.tensor_api.expand(x2ms_adapter.tensor_api.unsqueeze(box_a[:, 2:], 1), A, B, 2),
                       x2ms_adapter.tensor_api.expand(x2ms_adapter.tensor_api.unsqueeze(box_b[:, 2:], 0), A, B, 2))
    min_xy = x2ms_adapter.x2ms_max(x2ms_adapter.tensor_api.expand(x2ms_adapter.tensor_api.unsqueeze(box_a[:, :2], 1), A, B, 2),
                       x2ms_adapter.tensor_api.expand(x2ms_adapter.tensor_api.unsqueeze(box_b[:, :2], 0), A, B, 2))
    inter = x2ms_adapter.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = x2ms_adapter.tensor_api.expand_as(x2ms_adapter.tensor_api.unsqueeze(((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])), 1), inter)  # [A,B]
    area_b = x2ms_adapter.tensor_api.expand_as(x2ms_adapter.tensor_api.unsqueeze(((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])), 0), inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


# def reparameterize(mu, logvar):
#     std = x2ms_adapter.exp(0.5 * logvar)
#     # eps = torch.randn_like(std)
#     eps = mindspore.ops.randn_like(std)
#     return mu + eps * std


def kld(mu, logvar):
    mu = x2ms_adapter.tensor_api.view(mu, -1, x2ms_adapter.tensor_api.x2ms_size(mu, -1))
    logvar = x2ms_adapter.tensor_api.view(logvar, -1, x2ms_adapter.tensor_api.x2ms_size(logvar, -1))
    return (-0.5 * x2ms_adapter.x2ms_sum(1 + logvar - x2ms_adapter.tensor_api.x2ms_pow(mu, 2) - x2ms_adapter.tensor_api.exp(logvar))) / x2ms_adapter.tensor_api.x2ms_size(mu, 0)


def random_mask(size, prob):
    mask = x2ms_adapter.tensor_api.uniform_(x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.zeros(size, requires_grad=False))) < prob
    if x2ms_adapter.is_cuda_available():
        mask = mask
    return mask


def create_emb_layer(emb_matrix, non_trainable=True):
    print("start embedding")
    print(emb_matrix)
    num_embeddings, embedding_dim = x2ms_adapter.tensor_api.x2ms_size(emb_matrix)
    emb_layer = x2ms_nn.Embedding(num_embeddings, embedding_dim, padding_idx=0,_weight=emb_matrix)
    # emb_param = mindspore.Parameter(emb_matrix)
    print(emb_layer.embedding_table.asnumpy())
    # appenddict = {"embedding_table": emb_param}
    # mindspore.save_checkpoint(emb_layer, "glove.ckpt", append_dict=appenddict)
    # # print("save glove.ckpt completed")s
    # param_dict = mindspore.load_checkpoint("glove.ckpt")
    # mindspore.load_param_into_net(emb_layer, appenddict)   # 加载glove embedding
    if non_trainable:
        emb_layer.weight.requires_grad = False      # 如果不可训练，则将取消梯度
    print(emb_layer)
    return emb_layer


def load_embeddings(emb_text_filepath, id2vocab, emb_dim):
    print("start loading")
    vectors = bcolz.open(emb_text_filepath + '.dat')[:]
    words = pickle.load(open(emb_text_filepath + '.words.pkl', 'rb'))
    word2idx = pickle.load(open(emb_text_filepath + '.ids.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(id2vocab)
    emb_matrix = x2ms_adapter.zeros((matrix_len, emb_dim))
    words_found = 0

    for i in range(len(id2vocab)):
        print("loading "+str(i)+" id2vocab")
        word = id2vocab[i]
        try:
            emb_matrix[i] = x2ms_adapter.Tensor(glove[word])
            words_found += 1
        except KeyError:
            emb_matrix[i] = x2ms_adapter.Tensor(np.random.normal(scale=0.6, size=(emb_dim,)))
    print('word found : ', words_found)
    return emb_matrix       # Tensor shape(len(id2vocab), emb_dim) , 返回的是所有词的glove embedding


def prepare_embeddings(emb_text_filepath):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=emb_text_filepath + '.temp', mode='w')

    with open(emb_text_filepath, 'rb') as f:
        for l in f:
            line = x2ms_adapter.tensor_api.split(l.decode())
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((idx, -1)), rootdir=emb_text_filepath + '.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(emb_text_filepath + '.words.pkl', 'wb'))
    pickle.dump(word2idx, open(emb_text_filepath + '.ids.pkl', 'wb'))


def new_tensor(array, requires_grad=False):
    tensor = change_tensor(array, requires_grad=requires_grad)
    return tensor


def change_tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """
    Parameter 'device', 'pin_memory' are not supported.
    """
    # if mindspore.context.get_context('device_target') == 'Ascend' and result.dtype == mindspore.float64:
    #     result = mindspore.Tensor(data, dtype=mindspore.float32)
    # else:
    result = mindspore.Tensor(data,dtype=dtype)

    if not requires_grad:
        result = mindspore.ops.stop_gradient(result)
    return result

# def hotfix_pack_padded_sequence(input, lengths, batch_first=True):
#     lengths = torch.as_tensor(lengths, dtype=torch.int64)
#     lengths = lengths.cpu()
#     return PackedSequence(torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first))

def gru_forward(gru, input, lengths, state=None, batch_first=True):
    # gru.flatten_parameters()
    if  lengths.dtype != mindspore.float32 and lengths.dtype != mindspore.float16:
        lengths = lengths.astype(mindspore.float32)
    print(lengths)
    input_lengths, perm = mindspore.ops.Sort(descending=True)(lengths)

    input = input[perm]
    if state is not None:
        state = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.transpose(state[perm], 0, 1))

    total_length = x2ms_adapter.tensor_api.x2ms_size(input, 1)
    if not batch_first:
        input = x2ms_adapter.tensor_api.transpose(input, 0, 1)  # B x L x N -> L x B x N
    if input_lengths.dtype != mindspore.int32:
        input_lengths = input_lengths.astype(mindspore.int32)
    packed = x2ms_nn.pack_padded_sequence(input, input_lengths, batch_first)
    # packed = hotfix_pack_padded_sequence(embedded, input_lengths, batch_first)
    # self.gru.flatten_parameters()
    outputs, state = gru(packed, state)  # -> L x B x N * n_directions, 1, B, N
    outputs, output_lengths = x2ms_nn.pad_packed_sequence(outputs, batch_first=batch_first,
                                                                     total_length=total_length)  # unpack (back to padded)
    if  perm.dtype != mindspore.float32 and perm.dtype != mindspore.float16:
        perm = perm.astype(mindspore.float32)
    _, perm = mindspore.ops.Sort(descending=False)(perm)
    if not batch_first:
        outputs = x2ms_adapter.tensor_api.transpose(outputs, 0, 1)
    outputs = outputs[perm]
    state = x2ms_adapter.tensor_api.transpose(state, 0, 1)[perm]

    return outputs, state


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return x2ms_adapter.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = x2ms_adapter.tensor_api.x2ms_size(seq_q, 1)
    padding_mask = seq_k.ne(0)
    padding_mask = x2ms_adapter.tensor_api.expand(x2ms_adapter.tensor_api.unsqueeze(padding_mask, 1), -1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = x2ms_adapter.tensor_api.x2ms_size(seq)
    subsequent_mask = x2ms_adapter.triu(
        x2ms_adapter.ones((len_s, len_s), device=seq.device, dtype=mindspore.uint8), diagonal=1)
    if x2ms_adapter.is_cuda_available():
        subsequent_mask = subsequent_mask
    subsequent_mask = x2ms_adapter.tensor_api.expand(x2ms_adapter.tensor_api.unsqueeze(subsequent_mask, 0), sz_b, -1, -1)  # b x ls x ls
    subsequent_mask = 1 - subsequent_mask

    return subsequent_mask
