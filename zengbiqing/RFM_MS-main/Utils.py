import random
import time
import nltk
from Constants import *
from modules.Utils import *
# from transformers import BertTokenizer
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn_cell
from data.Utils import *

def get_ms():
    return time.time() * 1000


def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    np.random.seed(seed)
    mindspore.set_seed(seed)
    random.seed(seed)
    mindspore.set_seed(seed)
    mindspore.set_seed(seed)


def importance_sampling(prob, topk):
    m = mindspore.Tensor.random_categorical(logits=prob)
    indices = x2ms_adapter.tensor_api.transpose(m.sample((topk,)), 0, 1)  # batch, topk

    values = prob.gather(1, indices)
    return values, indices


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = x2ms_adapter.tensor_api.numel(lengths)
    max_len = max_len or x2ms_adapter.tensor_api.x2ms_max(lengths)
    mask = (x2ms_adapter.tensor_api.lt(x2ms_adapter.tensor_api.repeat(x2ms_adapter.tensor_api.type_as(x2ms_adapter.arange(0, max_len), lengths), batch_size, 1), x2ms_adapter.tensor_api.unsqueeze(lengths, 1)))
    if x2ms_adapter.is_cuda_available():
        mask = mask
    return mask


def start_end_mask(starts, ends, max_len):
    batch_size = len(starts)
    mask = x2ms_adapter.arange(1, max_len + 1)
    if x2ms_adapter.is_cuda_available():
        mask = mask
    mask = x2ms_adapter.tensor_api.expand(x2ms_adapter.tensor_api.unsqueeze(mask, 0), batch_size, -1)
    mask1 = mask >= x2ms_adapter.tensor_api.expand_as(x2ms_adapter.tensor_api.unsqueeze(starts, 1), mask)
    mask2 = mask <= x2ms_adapter.tensor_api.expand_as(x2ms_adapter.tensor_api.unsqueeze(ends, 1), mask)
    mask = (mask1 * mask2)
    return mask


# def decode_to_end(model, data, vocab2id, max_target_length=None, schedule_rate=1, softmax=False, encode_outputs=None,
#                   init_decoder_states=None, tgt=None, init_feedback_states=None):
#     # if tgt is None:
#     #     tgt = data['output']
#     # 查看回复
#     # print('train content: \n', tgt)
#     batch_size = len(data['id'])
#     if max_target_length is None:
#         max_target_length = x2ms_adapter.tensor_api.x2ms_size(tgt, 1)
#     if encode_outputs is None:
#         encode_outputs = model.encode(data)
#     if init_decoder_states is None:
#         init_decoder_states = model.init_decoder_states(data, encode_outputs)  # [batch_size, 1, hidden_size]
#         # print('init_decoder_states: ', init_decoder_states.size())
#     # feedback初始化
#     if init_feedback_states is None:
#         feedback_outputs = model.init_feedback_states(data, encode_outputs, init_decoder_states)
#
#     decoder_input = new_tensor([vocab2id[BOS_WORD]] * batch_size, requires_grad=False)  # 当前的前一个词
#
#     prob = x2ms_adapter.ones((batch_size,)) * schedule_rate
#     if x2ms_adapter.is_cuda_available():
#         prob = prob
#
#     all_gen_outputs = list()  # 存储pk, pv mix后的p
#     all_decode_outputs = [dict({'state': init_decoder_states})]  # 存储每个解码过程的pk, pv, state
#     all_feedback_states = list()  # 保存所有的feedback state
#     for t in range(max_target_length):
#         if t != 0:
#             all_decode_inputs = tgt[:, :t]  # 左闭右开，取到的是t之前所有的词
#             # 待完成：model.decoder_to_encoder()、完成feedback.forward()、修改model.decode()、测试、控制feedback维度
#             # feedback，输入为encoder输出（segment）和当前生成词前面的真值
#             feedback_outputs = model.decoder_to_encoder(data, encode_outputs, all_decode_inputs)  # all_decode_inputs用于GRU
#             all_feedback_states.append(feedback_outputs)
#             # decoder_outputs, decoder_states, ...
#             # 生成每个解码过程的pk, pv, state
#             decode_outputs = model.decode(
#                 data, decoder_input, encode_outputs, all_decode_outputs[-1], feedback_outputs
#             )
#             # 生成pk, pv mix后的p
#             output = model.generate(data, encode_outputs, decode_outputs, softmax=softmax)
#
#             all_decode_outputs.append(decode_outputs)
#             all_gen_outputs.append(output)
#
#         # 查看维度
#         # print('*' * 20, 'Decoder模块', '*' * 20)
#         # print('decode_outputs: ', decode_outputs)
#         # print('output: ', output)
#         # print('all_gen_outputs:', all_gen_outputs)
#         # print('all_decode_outputs: ', all_decode_outputs)
#
#         if schedule_rate >= 1:
#             decoder_input = tgt[:, t]  # decoder前一个词使用的是真实回复的前一个词
#         # elif schedule_rate <= 0:
#         #     probs, ids = model.to_word(data, output, 1)
#         #     decoder_input = model.generation_to_decoder_input(data, ids[:, 0])
#         # else:
#         #     probs, ids = model.to_word(data, output, 1)
#         #     indices = model.generation_to_decoder_input(data, ids[:, 0])
#         #
#         #     draws = x2ms_adapter.tensor_api.long(x2ms_adapter.bernoulli(prob))
#         #     decoder_input = tgt[:, t] * draws + indices * (1 - draws)
#
#     # all_gen_outputs = torch.cat(all_gen_outputs, dim=0).transpose(0, 1).contiguous()
#
#     return encode_outputs, all_gen_outputs


def randomk(gen_output, k=5, PAD=None, BOS=None, UNK=None):
    if PAD is not None:
        gen_output[:, PAD] = -float('inf')
    if BOS is not None:
        gen_output[:, BOS] = -float('inf')
    if UNK is not None:
        gen_output[:, UNK] = -float('inf')
    values, indices = importance_sampling(gen_output, k)
    # words=[[tgt_id2vocab[id.item()] for id in one] for one in indices]
    return values, indices


def topk(gen_output, k=5, PAD=None, BOS=None, UNK=None):
    if PAD is not None:
        gen_output[:, PAD] = 0
    if BOS is not None:
        gen_output[:, BOS] = 0
    if UNK is not None:
        gen_output[:, UNK] = 0
    if k > 1:
        values, indices = x2ms_adapter.topk(gen_output, k, dim=1, largest=True,
                                     sorted=True, out=None)
    else:
        values, indices = x2ms_adapter.x2ms_max(gen_output, dim=1, keepdim=True)
    return values, indices


def copy_topk(gen_output, vocab_map, vocab_overlap, k=5, PAD=None, BOS=None, UNK=None):
    vocab = gen_output[:, :x2ms_adapter.tensor_api.x2ms_size(vocab_map, -1)]
    dy_vocab = gen_output[:, x2ms_adapter.tensor_api.x2ms_size(vocab_map, -1):]

    vocab = vocab + x2ms_adapter.tensor_api.squeeze(x2ms_adapter.bmm(x2ms_adapter.tensor_api.unsqueeze(dy_vocab, 1), vocab_map), 1)
    dy_vocab = dy_vocab * vocab_overlap

    gen_output = x2ms_adapter.cat([vocab, dy_vocab], dim=-1)
    return topk(gen_output, k, PAD=PAD, BOS=BOS, UNK=UNK)


def remove_duplicate_once(sents, n=3):
    changed = False
    for b in range(len(sents)):
        sent = sents[b]
        if len(sent) <= n:
            continue

        for i in range(len(sent) - n):
            index = len(sent) - i - n
            if all(elem in sent[:index] for elem in sent[index:]):
                sents[b] = sent[:index]
                changed = True
                break
    return changed


def remove_duplicate(sents, n=3):
    changed = remove_duplicate_once(sents, n)
    while changed:
        changed = remove_duplicate_once(sents, n)


def to_sentence(batch_indices, id2vocab):
    batch_size = len(batch_indices)
    summ = list()
    for i in range(batch_size):
        indexes = batch_indices[i]
        text_summ2 = []
        for index in indexes:
            index = x2ms_adapter.tensor_api.item(index)
            w = id2vocab[index]
            if w == BOS_WORD or w == PAD_WORD:
                continue
            if w == EOS_WORD:
                break
            text_summ2.append(w)
        if len(text_summ2) == 0:
            text_summ2.append(UNK_WORD)
        summ.append(text_summ2)
    return summ


def to_copy_sentence(data, batch_indices, id2vocab, dyn_id2vocab_map):
    # 这是一段Python代码，它定义了一个名为to_copy_sentence的函数。该函数接受四个参数：data，batch_indices，id2vocab和dyn_id2vocab_map。
    # 该函数的主要目的是将输入数据转换为可读的文本摘要。
    # 函数首先从输入数据中提取出ID，然后对每个ID进行处理以生成文本摘要。
    # 最后，该函数返回一个列表，其中包含所有生成的文本摘要。
    ids = data['id']
    batch_size = len(batch_indices)
    summ = list()
    for i in range(batch_size):
        indexes = batch_indices[i]
        text_summ2 = []
        dyn_id2vocab = dyn_id2vocab_map[x2ms_adapter.tensor_api.item(ids[i])]
        for index in indexes:
            index = x2ms_adapter.tensor_api.item(index)
            if index < len(id2vocab):
                w = id2vocab[index]
            elif index - len(id2vocab) in dyn_id2vocab:
                w = dyn_id2vocab[index - len(id2vocab)]
            else:
                w = PAD_WORD

            if w == BOS_WORD or w == PAD_WORD:
                continue

            if w == EOS_WORD:
                break

            text_summ2.append(w)

        if len(text_summ2) == 0:
            text_summ2.append(UNK_WORD)

        summ.append(text_summ2)
    return summ



#
#

