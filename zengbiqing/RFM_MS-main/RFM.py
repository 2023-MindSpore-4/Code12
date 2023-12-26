import numpy

from EncDecModel import *
from modules.BilinearAttention import *
from modules.Highway import *
from data.Utils import *
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_cell
import x2ms_adapter.nn_functional


class GenEncoder(nn.Cell):
    # 它的功能是定义一个生成式编码器，用于将多个输入序列编码为隐藏状态
    # 这个类的参数如下：
    #
    # n: 一个整数，表示输入序列的个数。
    # src_vocab_size: 一个整数，表示输入序列的词汇表大小。
    # embedding_size: 一个整数，表示词嵌入的维度。
    # hidden_size: 一个整数，表示隐藏状态的维度。
    # emb_matrix: 一个可选的张量，表示预训练的词嵌入矩阵。如果为None，则使用随机初始化的词嵌入。
    # 这个类的属性如下：
    #
    # self.n: 保存输入序列个数的属性。
    # self.c_embedding: 一个模块列表，包含n个词嵌入层，用于将输入序列转换为词向量。如果提供了emb_matrix，
    # 则使用create_emb_layer函数创建词嵌入层，否则使用x2ms_nn.Embedding创建词嵌入层。x2ms_nn是MindSpore框架中的神经网络模块。
    # self.c_encs: 一个模块列表，包含n个GRU层，用于将词向量编码为隐藏状态。GRU是一种循环神经网络单元，可以处理变长的序列数据。
    # 每个GRU层都是双向的，即同时从前向后和从后向前处理序列，并将两个方向的输出拼接起来。
    # 第一个GRU层的输入维度为embedding_size，其他GRU层的输入维度为embedding_size + hidden_size，因为要将上一层的输出作为附加信息传递给下一层。
    # 每个GRU层的输出维度为hidden_size / 2，因为要将两个方向的输出拼接起来。

    def __init__(self, n, src_vocab_size, embedding_size, hidden_size, emb_matrix=None):
        # 构造方法，用于初始化类的属性和参数。
        super(GenEncoder, self).__init__()
        self.n = n

        if emb_matrix is None:
            self.c_embedding = x2ms_nn.ModuleList(
                [x2ms_nn.Embedding(src_vocab_size, embedding_size, padding_idx=0) for i in range(n)])
        else:
            self.c_embedding = x2ms_nn.ModuleList(
                [create_emb_layer(emb_matrix) for i in range(n)])  # [1, all_word_num, emb_dim]
        self.c_encs = x2ms_nn.ModuleList([x2ms_nn.GRU(embedding_size, int(hidden_size / 2), num_layers=1, bidirectional=True,
                                            batch_first=True) if i == 0 else x2ms_nn.GRU(embedding_size + hidden_size,
                                                                                    int(hidden_size / 2), num_layers=1,
                                                                                    bidirectional=True,
                                                                                    batch_first=True) for i in
                                     range(n)])

    def construct(self, c):  # [batch_size, word_num]    word_num: knowledge_len(300) or context_len(65)
        # 构造方法，用于定义类的计算逻辑。x是一个列表，包含n个输入序列。
        # 每个输入序列都是一个形状为(batch_size, seq_len)的张量，其中batch_size是批量大小，seq_len是序列长度。
        # 这个方法首先对每个输入序列进行词嵌入和GRU编码，并将每一层的输出保存在一个列表中。然后返回这个列表作为最终的输出。
        c_outputs = []
        c_states = []

        c_mask = x2ms_adapter.tensor_api.detach(mindspore.ops.ne(c,0))  # [batch_size, word_num]
        # c_lengths = x2ms_adapter.tensor_api.detach(x2ms_adapter.x2ms_sum(c_mask, dim=1))  # [batch_size]
        lengths = c.shape[1]
        c_lengths = mindspore.Tensor([lengths])
        # c_emb = x2ms_adapter.nn_functional.dropout(self.c_embedding[0](c), training=self.training)
        c_emb = self.c_embedding[0](c)
        print("c_emb.shape:")
        print(c_emb.shape)
        c_enc_output = c_emb  # [batch_size, word_num, emb_dim]   word_num: knowledge_len(300) or context_len(65)

        for i in range(self.n):  # self.n == 1
            if i > 0:
                # c_enc_output = x2ms_adapter.cat([c_enc_output, x2ms_adapter.nn_functional.dropout(self.c_embedding[i](c), training=self.training)],
                #                          dim=-1)
                c_enc_output = x2ms_adapter.cat([c_enc_output, self.c_embedding[i](c)], dim=-1)
            print("c_enc_output.shape:")
            print(c_enc_output.shape)
            c_enc_output, c_state = gru_forward(self.c_encs[i], c_enc_output, c_lengths)
            print("after gru_forward,c_enc_output.shape:")
            print(c_enc_output.shape)
            print("after gru_forward,c_state.shape:")
            print(c_state.shape)
            c_outputs.append(x2ms_adapter.tensor_api.unsqueeze(c_enc_output, 1))
            c_states.append(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.view(c_state, x2ms_adapter.tensor_api.x2ms_size(c_state, 0), -1), 1))

        output = x2ms_adapter.cat(c_outputs, dim=1)
        print("gecencoder output:")
        print(output.shape)
        state = x2ms_adapter.cat(c_states,dim=1)
        print("gecencoder state:")
        print(state.shape)
        return output, state   # [batch_size, 1, word_num, hidden_size] [batch_size, 1, hidden_size]


class KnowledgeSelector(nn.Cell):
    # 它的功能是定义一个知识选择器，用于从多个知识源中选择最相关的知识。这个类的参数如下：
    #
    # hidden_size: 一个整数，表示隐藏状态的维度。
    # min_window_size: 一个整数，表示知识源中最小的窗口大小。
    # n_windows: 一个整数，表示知识源中窗口的个数。
    # 这个类的属性如下：
    #
    # self.min_window_size: 保存最小窗口大小的属性。
    # self.n_windows: 保存窗口个数的属性。
    # self.b_highway: 一个高速公路网络，用于对背景知识进行非线性变换。高速公路网络是一种具有门控机制的全连接层，可以控制信息的流动。
    # self.c_highway: 一个高速公路网络，用于对候选知识进行非线性变换。
    # self.match_attn: 一个双线性注意力模块，用于计算背景知识和候选知识之间的匹配程度。双线性注意力是一种利用两个输入向量之间的内积来计算注意力权重的方法。
    # self.area_attn: 一个双线性注意力模块，用于计算候选知识中不同区域（窗口）的重要性。

    def __init__(self, hidden_size, min_window_size=5, n_windows=4):
        # 构造方法，用于初始化类的属性和参数。
        super(KnowledgeSelector, self).__init__()
        self.min_window_size = min_window_size
        self.n_windows = n_windows

        self.b_highway = Highway(hidden_size * 2, hidden_size * 2, num_layers=2)
        self.c_highway = Highway(hidden_size * 2, hidden_size * 2, num_layers=2)
        self.match_attn = BilinearAttention(query_size=hidden_size * 2, key_size=hidden_size * 2,
                                            hidden_size=hidden_size * 2)
        self.area_attn = BilinearAttention(query_size=hidden_size, key_size=hidden_size, hidden_size=hidden_size)

    def match(self, b_enc_output, c_enc_output, c_state, b_mask, c_mask):
        # 匹配方法，用于计算背景知识和候选知识之间的匹配分数。这个方法的参数如下：
        # b_enc_output: 一个张量，表示背景知识经过编码后的输出，形状为(batch_size, b_seq_len, hidden_size)。
        # c_enc_output: 一个张量，表示候选知识经过编码后的输出，形状为(batch_size, c_seq_len, hidden_size)。
        # c_state: 一个张量，表示候选知识经过编码后的最终状态，形状为(batch_size, hidden_size)。
        # b_mask: 一个张量，表示背景知识中哪些位置是有效的（非填充），形状为(batch_size, b_seq_len)。
        # c_mask: 一个张量，表示候选知识中哪些位置是有效的（非填充），形状为(batch_size, c_seq_len)。
        # 这个方法首先将背景知识和候选知识分别通过高速公路网络进行变换，并将候选知识的最终状态拼接到每个位置上。然后使用双线性注意力模块计算两者之间的匹配矩阵。
        # 接着对匹配矩阵进行掩码处理，将无效位置（填充）置为负无穷大或零。最后在候选知识维度上取最大值，得到每个背景知识位置对应的匹配分数。返回这个分数向量作为输出。
        b_enc_output = self.b_highway(x2ms_adapter.cat([b_enc_output, x2ms_adapter.tensor_api.expand(c_state, -1, x2ms_adapter.tensor_api.x2ms_size(b_enc_output, 1), -1)], dim=-1))
        c_enc_output = self.c_highway(x2ms_adapter.cat([c_enc_output, x2ms_adapter.tensor_api.expand(c_state, -1, x2ms_adapter.tensor_api.x2ms_size(c_enc_output, 1), -1)], dim=-1))

        matching = self.match_attn.matching(b_enc_output, c_enc_output)
        print("this is matching.shape:")
        print(matching.shape)
        print(matching)
        matching = x2ms_adapter.tensor_api.masked_fill(matching, x2ms_adapter.tensor_api.unsqueeze(c_mask, 1), -float('inf'))
        matching = x2ms_adapter.tensor_api.masked_fill(matching, x2ms_adapter.tensor_api.unsqueeze(b_mask, 2), 0.)
        print(matching)
        score = x2ms_adapter.tensor_api.x2ms_max(matching, dim=-1)[0]

        return score

    def segments(self, b_enc_output, b_score, c_state):
        # 分段方法，用于将背景知识划分为不同大小的窗口，并计算每个窗口对应的区域注意力权重。这个方法的参数如下：
        # b_enc_output: 一个张量，表示背景知识经过编码后的输出，形状为(batch_size, b_seq_len, hidden_size)。
        # b_score: 一个张量，表示背景知识和候选知识之间的匹配分数向量，形状为(batch_size, b_seq_len)。
        # c_state: 一个张量，表示候选知识经过编码后的最终状态，形状为(batch_size, hidden_size)。
        # 这个方法首先初始化窗口大小为最小窗口大小，然后遍历窗口个数，对背景知识进行滑动窗口切分，得到每个窗口对应的子序列。
        # 然后将每个子序列和候选知识的最终状态作为输入，使用双线性注意力模块计算每个窗口的区域注意力权重。
        # 同时，对匹配分数向量也进行滑动窗口切分，得到每个窗口对应的匹配分数和。
        # 将每个窗口的区域注意力权重和匹配分数和分别保存在一个列表中。
        # 最后将这两个列表拼接起来，得到每个窗口的区域注意力权重向量和匹配分数向量。
        # 返回这两个向量作为输出。
        window_size = self.min_window_size
        bs = list()
        ss = list()
        for i in range(self.n_windows):
            # b = b_enc_output.unfold(1, window_size, self.min_window_size)
            b = bunfold(b_enc_output,1, window_size, self.min_window_size)
            print("after bunfold the shape is:")
            print(b.shape)
            b = x2ms_adapter.tensor_api.contiguous(b) # 输入到双线性的b应该是（1,64,4,256）(以knowleadge_length为256，hidden_size为256，min_windowsizew为4为例，这些默认参数不可变动否则会出错)
            b = x2ms_adapter.tensor_api.squeeze(self.area_attn(x2ms_adapter.tensor_api.unsqueeze(c_state, 1), b, b)[0], 2)
            bs.append(b)

            # s = b_score.unfold(1, window_size, self.min_window_size)
            s = cunfold(b_score,1, window_size, self.min_window_size)
            s = x2ms_adapter.x2ms_sum(s,dim=-1)
            ss.append(s)

            window_size += self.min_window_size
        return x2ms_adapter.cat(bs, dim=1), x2ms_adapter.cat(ss, dim=1)

    def construct(self, b_enc_output, c_enc_output, c_state, b_mask, c_mask):
        # 构造方法，用于定义类的计算逻辑。这个方法的参数如下：
        # b_enc_output: 一个张量，表示背景知识经过编码后的输出，形状为(batch_size, b_seq_len, hidden_size)。
        # c_enc_output: 一个张量，表示候选知识经过编码后的输出，形状为(batch_size, c_seq_len, hidden_size)。
        # c_state: 一个张量，表示候选知识经过编码后的最终状态，形状为(batch_size, hidden_size)。
        # b_mask: 一个张量，表示背景知识中哪些位置是有效的（非填充），形状为(batch_size, b_seq_len)。
        # c_mask: 一个张量，表示候选知识中哪些位置是有效的（非填充），形状为(batch_size, c_seq_len)。
        # 这个方法首先调用match方法，计算背景知识和候选知识之间的匹配分数向量。
        # 然后调用segments方法，计算每个窗口的区域注意力权重向量和匹配分数向量。
        # 接着对匹配分数向量进行softmax归一化，得到每个窗口的选择概率。
        # 最后使用矩阵乘法，将选择概率和区域注意力权重向量相乘，得到最终选择的背景知识子序列。
        # 返回这个子序列、选择概率和匹配分数作为输出。
        b_score = self.match(b_enc_output, c_enc_output, c_state, b_mask, c_mask)  # [batch_size, knowledge_len]
        segments, s_score = self.segments(b_enc_output, b_score, c_state)

        s_score = x2ms_adapter.nn_functional.softmax(s_score, dim=-1)  # [batch_size, knowledge_len/4]

        segments = x2ms_adapter.bmm(x2ms_adapter.tensor_api.unsqueeze(s_score, 1), segments)  # [batch_size, 1, hidden_size]

        return segments, s_score, b_score
        # [batch_size, 1, hidden_size], [batch_size, knowledge_len/window_size(4)], [batch_size, knowledge_len]


def bunfold(b,dim, window_size, min_window_size):
    # pytorch的torch.tensor.unfold的粗略实现
    length = b.shape[dim]
    print("this is unfold input b.shape[dim]")
    print(length)
    result = []
    for i in range(0, length, min_window_size):
        result.append(b[:, i:i + window_size, :])
        print("this is unfold slite.shape")
        print(b[:, i:i + window_size, :].shape)
    y = mindspore.ops.stack(result, axis=dim)
    print("this is unfold shape")
    print(y.shape)
    return y


def cunfold(c,dim, window_size, min_window_size):
    # pytorch的torch.tensor.unfold的粗略实现
    length = c.shape[dim]
    print("this is cunfold input c.shape[dim]")
    print(length)
    result = []
    for i in range(0, length, min_window_size):
        result.append(c[:, i:i + window_size])
        print("this is cunfold slite.shape")
        print(c[:, i:i + window_size].shape)
    y = mindspore.ops.stack(result, axis=dim)
    print("this is cunfold shape")
    print(y.shape)
    return y

class CopyGenerator(nn.Cell):
    # 定义一个复制生成器，用于从背景知识中复制单词到生成的文本中。这个类的参数如下：
    #
    # embedding_size: 一个整数，表示词嵌入的维度。
    # hidden_size: 一个整数，表示隐藏状态的维度。
    # knowledge_len: 一个整数，表示知识源中窗口的个数。
    # 这个类的属性如下：
    #
    # self.linear: 一个线性层，用于将知识源中每个窗口的区域注意力权重向量转换为隐藏状态维度。
    # self.cat_linear: 一个线性层，用于将知识源中每个窗口的区域注意力权重向量和候选知识子序列拼接后进行非线性变换。
    # self.b_attn: 一个双线性注意力模块，用于计算生成的单词和背景知识之间的复制概率。双线性注意力是一种利用两个输入向量之间的内积来计算注意力权重的方法。
    def __init__(self, embedding_size, hidden_size, knowledge_len):
        super(CopyGenerator, self).__init__()
        self.linear = x2ms_nn.Linear(knowledge_len, hidden_size)
        self.cat_linear = x2ms_nn.Linear(hidden_size * 2, hidden_size)
        self.b_attn = BilinearAttention(query_size=embedding_size + hidden_size * 2, key_size=hidden_size,
                                        hidden_size=hidden_size)

    def construct(self, p, word, state, feedback_states, segment, b_enc_output, c_enc_output, b_mask, c_mask):
        # 构造方法，用于定义类的计算逻辑。这个方法的参数如下：
        # p: 一个张量，表示生成文本中每个位置对应的单词概率分布，形状为(batch_size, vocab_size)。
        # word: 一个张量，表示生成文本中当前位置对应的单词向量，形状为(batch_size, embedding_size)。
        # state: 一个张量，表示生成文本中当前位置对应的隐藏状态向量，形状为(batch_size, hidden_size)。
        # feedback_states: 一个张量，表示知识源中每个窗口对应的选择概率向量，形状为(batch_size, knowledge_len)。
        # segment: 一个张量，表示知识源中最终选择的候选知识子序列向量，形状为(batch_size, hidden_size)。
        # b_enc_output: 一个张量，表示背景知识经过编码后的输出，形状为(batch_size, b_seq_len, hidden_size)。
        # c_enc_output: 一个张量，表示候选知识经过编码后的输出，形状为(batch_size, c_seq_len, hidden_size)。
        # b_mask: 一个张量，表示背景知识中哪些位置是有效的（非填充），形状为(batch_size, b_seq_len)。
        # c_mask: 一个张量，表示候选知识中哪些位置是有效的（非填充），形状为(batch_size, c_seq_len)。
        # 这个方法首先将知识源中每个窗口的选择概率向量通过线性层转换为隐藏状态维度。
        # 然后将转换后的向量和候选知识子序列向量拼接起来，并通过另一个线性层进行非线性变换。
        # 接着将生成文本中当前位置对应的单词向量、隐藏状态向量和非线性变换后的向量拼接起来，并作为查询向量输入到双线性注意力模块中。
        # 最后使用双线性注意力模块计算生成文本中当前位置对应的单词和背景知识之间的复制概率，并返回这个概率向量作为输出。
        feedback_states = self.linear(feedback_states)
        segment_mix = self.cat_linear(x2ms_adapter.cat([feedback_states, segment], dim=-1))
        p = x2ms_adapter.tensor_api.squeeze(self.b_attn.score(x2ms_adapter.cat([word, state, segment_mix], dim=-1), b_enc_output,
                              mask=x2ms_adapter.tensor_api.unsqueeze(b_mask, 1)), 1)
        return p


class VocabGenerator(nn.Cell):
    def __init__(self, embedding_size, hidden_size, knowledge_len, vocab_size):
        # 定义一个词汇生成器，用于从词汇表中生成单词到生成的文本中。这个类的参数如下：
        #
        # embedding_size: 一个整数，表示词嵌入的维度。
        # hidden_size: 一个整数，表示隐藏状态的维度。
        # knowledge_len: 一个整数，表示知识源中窗口的个数。
        # vocab_size: 一个整数，表示词汇表的大小。
        super(VocabGenerator, self).__init__()
        self.linear = x2ms_nn.Linear(knowledge_len, hidden_size)
        self.cat_linear = x2ms_nn.Linear(hidden_size * 2, hidden_size)

        self.c_attn = BilinearAttention(query_size=embedding_size + hidden_size * 2, key_size=hidden_size,
                                        hidden_size=hidden_size)
        self.b_attn = BilinearAttention(query_size=embedding_size + hidden_size * 2, key_size=hidden_size,
                                        hidden_size=hidden_size)

        self.readout = x2ms_nn.Linear(embedding_size + 4 * hidden_size, hidden_size)
        self.generator = x2ms_nn.Linear(hidden_size, vocab_size)

    def construct(self, p, word, state, feedback_states, segment, b_enc_output, c_enc_output, b_mask, c_mask):
        # 构造方法，用于定义类的计算逻辑。这个方法的参数如下：
        # p: 一个张量，表示生成文本中每个位置对应的单词概率分布，形状为(batch_size, vocab_size)。
        # word: 一个张量，表示生成文本中当前位置对应的单词向量，形状为(batch_size, embedding_size)。
        # state: 一个张量，表示生成文本中当前位置对应的隐藏状态向量，形状为(batch_size, hidden_size)。
        # feedback_states: 一个张量，表示知识源中每个窗口对应的选择概率向量，形状为(batch_size, knowledge_len)。
        # segment: 一个张量，表示知识源中最终选择的候选知识子序列向量，形状为(batch_size, hidden_size)。
        # b_enc_output: 一个张量，表示背景知识经过编码后的输出，形状为(batch_size, b_seq_len, hidden_size)。
        # c_enc_output: 一个张量，表示候选知识经过编码后的输出，形状为(batch_size, c_seq_len, hidden_size)。
        # b_mask: 一个张量，表示背景知识中哪些位置是有效的（非填充），形状为(batch_size, b_seq_len)。
        # c_mask: 一个张量，表示候选知识中哪些位置是有效的（非填充），形状为(batch_size, c_seq_len)。
        # 这个方法首先将知识源中每个窗口的选择概率向量通过线性层转换为隐藏状态维度。
        # 然后将转换后的向量和候选知识子序列向量拼接起来，并通过另一个线性层进行非线性变换。
        # 接着将生成文本中当前位置对应的单词向量、隐藏状态向量和非线性变换后的向量拼接起来，并作为查询向量输入到两个双线性注意力模块中。
        # 分别使用双线性注意力模块计算生成文本中当前位置对应的单词和候选知识、背景知识之间的上下文向量，并将这两个向量分别保存在一个列表中。
        # 最后将这个列表拼接起来，得到每个窗口的区域注意力权重向量。返回这个向量作为输出。
        feedback_states = self.linear(feedback_states)
        segment_mix = self.cat_linear(x2ms_adapter.cat([feedback_states, segment], dim=-1))

        c_output, _ = self.c_attn(x2ms_adapter.cat([word, state, segment_mix], dim=-1), c_enc_output, c_enc_output,
                                  mask=x2ms_adapter.tensor_api.unsqueeze(c_mask, 1))
        c_output = x2ms_adapter.tensor_api.squeeze(c_output, 1)

        b_output, _ = self.b_attn(x2ms_adapter.cat([word, state, segment_mix], dim=-1), b_enc_output, b_enc_output,
                                  mask=x2ms_adapter.tensor_api.unsqueeze(b_mask, 1))
        b_output = x2ms_adapter.tensor_api.squeeze(b_output, 1)

        concat_output = x2ms_adapter.cat((x2ms_adapter.tensor_api.squeeze(word, 1), x2ms_adapter.tensor_api.squeeze(state, 1), x2ms_adapter.tensor_api.squeeze(segment_mix, 1), c_output, b_output),
                                  dim=-1)

        feature_output = self.readout(concat_output)

        p = x2ms_adapter.nn_functional.softmax(self.generator(feature_output), dim=-1)

        return p


class StateTracker(nn.Cell):
    def __init__(self, embedding_size, hidden_size):
        super(StateTracker, self).__init__()

        self.linear = x2ms_nn.Linear(hidden_size * 2, hidden_size)
        self.gru = x2ms_nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)

    def initialize(self, segment, state):
        return self.linear(x2ms_adapter.cat([state, segment], dim=-1))

    def construct(self, word, state):
        return x2ms_adapter.tensor_api.transpose(self.gru(word, x2ms_adapter.tensor_api.transpose(state, 0, 1))[1], 0, 1)


class Mixturer(nn.Cell):
    # 定义一个混合器，用于将两个概率分布进行加权混合。这个类的参数如下：hidden_size: 一个整数，表示隐藏状态的维度。
    # 这个类的属性如下：self.linear1: 一个线性层，用于将隐藏状态向量映射到一个标量，表示混合权重。
    def __init__(self, hidden_size):
        super(Mixturer, self).__init__()
        self.linear1 = x2ms_nn.Linear(hidden_size, 1)

    def construct(self, state, dists1, dists2, dyn_map):
        # decode_outputs['state'][1, 1, 256], decode_outputs['p_v'][1,59222(item_size即vocab_size)], decode_outputs['p_k'][1,256], data['dyn_map']
        # def construct(self, state, dists1, dists2, dyn_map): 构造方法，用于定义类的计算逻辑。这个方法的参数如下：
        # state: 一个张量，表示当前位置对应的隐藏状态向量，形状为(batch_size, 1, hidden_size)。
        # dists1: 一个张量，表示第一个概率分布向量，形状为(batch_size, vocab_size)。
        # dists2: 一个张量，表示第二个概率分布向量，形状为(batch_size, vocab_size)。
        # dyn_map: 一个张量，表示动态词汇表映射矩阵，形状为(batch_size, vocab_size, vocab_size)。
        # 这个方法首先通过线性层将隐藏状态向量映射到一个标量，并通过sigmoid函数将其转换为混合权重。
        # 然后将第二个概率分布向量和动态词汇表映射矩阵相乘，得到调整后的概率分布向量。
        # 接着将两个概率分布向量按照混合权重进行加权平均，得到最终的概率分布向量。
        # 返回这个向量作为输出。
        # 最后的p的shape为（1,vocab_size+经过buildmap之后的dyn_map中的max）
        p_k_v = x2ms_adapter.sigmoid(self.linear1(x2ms_adapter.tensor_api.squeeze(state, 1)))

        dists2 = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.bmm(x2ms_adapter.tensor_api.unsqueeze(dists2, 1), dyn_map), 1)

        dist = x2ms_adapter.cat([p_k_v * dists1, (1. - p_k_v) * dists2], dim=-1)

        return dist


# feedback mechanism
class Feedback(nn.Cell):
    def __init__(self, embedding_size, knowledge_len, context_len, hidden_size):
        super(Feedback, self).__init__()

        self.b_linear = x2ms_nn.Linear(hidden_size * 2, knowledge_len)
        self.b_gru = x2ms_nn.GRU(embedding_size, knowledge_len, num_layers=1, bidirectional=False, batch_first=True)

    def initialize(self, segment, state):
        b_init = self.b_linear(x2ms_adapter.cat([state, segment], dim=-1))  # [batch_size, 1, knowledge_len]
        return b_init

    def construct(self, word_emb, b_states):
        b_state = x2ms_adapter.tensor_api.transpose(self.b_gru(word_emb, x2ms_adapter.tensor_api.transpose(b_states, 0, 1))[1], 0, 1)  # [batch_size, 1, knowledge_len]
        return b_state


class Criterion(object):
    # 定义一个损失函数，用于计算生成的文本和参考文本之间的负对数似然损失。这个类的参数如下：
    #
    # tgt_vocab_size: 一个整数，表示目标词汇表的大小。
    # eps: 一个浮点数，表示一个很小的正数，用于避免对零取对数。
    # 这个类的属性如下：
    #
    # self.eps: 保存eps参数的属性。
    # self.offset: 一个整数，表示动态词汇表在生成输出中的偏移量，等于目标词汇表的大小。
    def __init__(self, tgt_vocab_size, eps=1e-10):
        super(Criterion, self).__init__()
        self.eps = eps
        self.offset = tgt_vocab_size

    def __call__(self, gen_output, response, dyn_response, UNK, reduction='mean'):
        # 调用方法，用于计算损失值。这个方法的参数如下：
        # gen_output: 一个张量，表示生成文本中每个位置对应的单词概率分布。
        # response: 一个张量，表示参考文本中每个位置对应的单词索引，形状为(batch_size, seq_len)。其中seq_len是序列长度。
        # dyn_response: 一个张量，表示参考文本中每个位置对应的动态词汇表中的单词索引，形状为(batch_size, seq_len)。
        # UNK: 一个整数，表示未知单词在目标词汇表中的索引。
        # reduction: 一个字符串，表示损失值的聚合方式。可以是’mean’或’none’。
        # 如果是’mean’，则返回所有位置上损失值的平均值；
        # 如果是’none’，则返回每个位置上的损失值向量。
        # 这个方法首先判断生成输出是否有多余两个维度，
        # 如果有，则将其展平为二维张量。
        # 然后根据参考文本和动态参考文本在生成输出中获取对应位置上的概率值，并乘以相应的掩码向量。
        # 其中掩码向量用于过滤掉填充位置和未知单词位置上的概率值。
        # 接着将两个概率值相加，并加上一个很小的正数eps，然后取对数。
        # 最后将对数概率值乘以另一个掩码向量，用于过滤掉填充位置上的损失值，并根据reduction参数返回相应的损失值。

        dyn_not_pad = x2ms_adapter.tensor_api.x2ms_float(mindspore.ops.ne(dyn_response,0))
        v_not_unk = x2ms_adapter.tensor_api.x2ms_float(mindspore.ops.ne(response,UNK))
        v_not_pad = x2ms_adapter.tensor_api.x2ms_float(mindspore.ops.ne(response,0))

        if len(x2ms_adapter.tensor_api.x2ms_size(gen_output)) > 2:
            gen_output = x2ms_adapter.tensor_api.view(gen_output, -1, x2ms_adapter.tensor_api.x2ms_size(gen_output, -1))
        if gen_output.shape[0] != response.shape[1]:
            pad_length = response.shape[1] - gen_output.shape[0]
            paddings = ((0, pad_length), (0, 0))
            gen_output = mindspore.ops.Pad(paddings)(gen_output)
        print("this is new gen_output.shape")
        print(gen_output.shape)
        p_dyn = x2ms_adapter.tensor_api.view(mindspore.ops.gather_d(gen_output, 1, x2ms_adapter.tensor_api.view(dyn_response, -1, 1) + self.offset),-1)
        print("this is p_dyn.shape")
        print(p_dyn.shape)
        p_dyn = x2ms_adapter.tensor_api.mul(p_dyn, x2ms_adapter.tensor_api.view(dyn_not_pad, -1))

        p_v = x2ms_adapter.tensor_api.view(mindspore.ops.gather_d(gen_output, 1, x2ms_adapter.tensor_api.view(response, -1, 1) ), -1)
        p_v = x2ms_adapter.tensor_api.mul(p_v, x2ms_adapter.tensor_api.view(v_not_unk, -1))

        p = p_dyn + p_v + self.eps
        p = x2ms_adapter.tensor_api.log(p)

        loss = -x2ms_adapter.tensor_api.mul(p, x2ms_adapter.tensor_api.view(v_not_pad, -1))
        if reduction == 'mean':
            return x2ms_adapter.x2ms_sum(loss) /  x2ms_adapter.x2ms_sum(v_not_pad)
        elif reduction == 'none':
            return x2ms_adapter.tensor_api.view(loss, x2ms_adapter.tensor_api.x2ms_size(response))


class RFM(EncDecModel):
    # 定义一个基于编码器-解码器模型的响应生成器，用于根据背景知识和候选知识生成对话响应。
    # 这个类的参数如下：
    # min_window_size: 一个整数，表示知识源中最小的窗口大小。
    # num_windows: 一个整数，表示知识源中窗口的个数。
    # embedding_size: 一个整数，表示词嵌入的维度。
    # knowledge_len: 一个整数，表示知识源中每个窗口的区域注意力权重向量的维度。
    # context_len: 一个整数，表示对话上下文的长度。
    # hidden_size: 一个整数，表示隐藏状态的维度。
    # vocab2id: 一个字典，表示词汇表到索引的映射。
    # id2vocab: 一个字典，表示索引到词汇表的映射。
    # max_dec_len: 一个整数，表示生成响应的最大长度。
    # beam_width: 一个整数，表示束搜索的宽度。
    # emb_matrix: 一个张量，表示预训练的词嵌入矩阵。如果为None，则随机初始化。
    # eps: 一个浮点数，表示一个很小的正数，用于避免对零取对数。
    #
    # 这个类的属性如下：
    # self.vocab_size: 一个整数，表示词汇表的大小。
    # self.vocab2id: 保存vocab2id参数的属性。
    # self.id2vocab: 保存id2vocab参数的属性。
    # self.b_encoder: 一个编码器对象，用于对背景知识进行编码。
    # self.c_encoder: 一个编码器对象，用于对候选知识进行编码。
    # self.embedding: 一个嵌入层对象，用于将输入单词转换为向量。如果提供了emb_matrix参数，则使用预训练的词嵌入；否则随机初始化。
    # self.feedback_embedding: 一个嵌入层对象，用于将反馈单词转换为向量。如果提供了emb_matrix参数，则使用预训练的词嵌入；否则随机初始化。
    # self.state_tracker: 一个状态跟踪器对象，用于根据对话上下文和候选知识更新解码器的初始状态。
    # self.k_selector: 一个知识选择器对象，用于从多个知识源中选择最相关的知识子序列和区域注意力权重向量。
    # self.c_generator: 一个复制生成器对象，用于从背景知识中复制单词到生成响应中。
    # self.v_generator: 一个词汇生成器对象，用于从词汇表中生成单词到生成响应中。
    # self.mixture: 一个混合器对象，用于将复制概率分布和词汇概率分布进行加权混合，得到最终的单词概率分布。
    # self.criterion: 一个损失函数对象，用于计算生成响应和参考响应之间的负对数似然损失。
    # self.feedback: 一个反馈对象，用于根据生成响应和参考响应提供反馈信息给状态跟踪器和知识选择器。
    # self.segment_linear: 一个线性层对象，用于将知识源中每个窗口的区域注意力权重向量和匹配分数向量拼接后进行非线性变换。
    def __init__(self, min_window_size, num_windows, embedding_size, knowledge_len, context_len, hidden_size, vocab2id,
                 id2vocab, max_dec_len,
                 beam_width, emb_matrix=None, eps=1e-10):
        super(RFM, self).__init__(vocab2id=vocab2id, max_dec_len=max_dec_len, beam_width=beam_width, eps=eps)
        self.vocab_size = len(vocab2id)
        self.vocab2id = vocab2id
        self.id2vocab = id2vocab
        # b和c两个encoder在encode函数中调用
        self.b_encoder = GenEncoder(1, self.vocab_size, embedding_size, hidden_size, emb_matrix=emb_matrix)
        self.c_encoder = GenEncoder(1, self.vocab_size, embedding_size, hidden_size, emb_matrix=emb_matrix)
        # embedding在decode函数中调用
        if emb_matrix is None:
            self.embedding = x2ms_nn.Embedding(self.vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = create_emb_layer(emb_matrix)  # 得到glove embedding

        if emb_matrix is None:
            self.feedback_embedding = x2ms_nn.Embedding(self.vocab_size, embedding_size, padding_idx=0)
        else:
            self.feedback_embedding = create_emb_layer(emb_matrix)  # 得到glove embedding
        # init_decoder_states调用，decode to end调用
        self.state_tracker = StateTracker(embedding_size, hidden_size)
        # 以下三个在encode中调用
        self.k_selector = KnowledgeSelector(hidden_size, min_window_size=min_window_size, n_windows=num_windows)
        self.c_generator = CopyGenerator(embedding_size, hidden_size, knowledge_len)
        self.v_generator = VocabGenerator(embedding_size, hidden_size, knowledge_len, self.vocab_size)
        # generate调用，decode to end调用
        self.mixture = Mixturer(hidden_size)
        # 计算loss中调用
        self.criterion = Criterion(self.vocab_size)
        # init_feedback_states中调用，decode to end中调用
        self.feedback = Feedback(embedding_size, knowledge_len, context_len, hidden_size)
        # encode中调用，
        self.segment_linear = x2ms_nn.Linear(int(knowledge_len / min_window_size + knowledge_len), hidden_size)

    # decode_to_end调用
    def encode(self, data):
        # 对输入数据进行编码，返回编码后的输出和状态。这个方法的参数如下：
        # data: 一个字典，表示输入数据，包含以下键值对：
        # ‘unstructured_knowledge’: 一个张量，表示背景知识中每个位置对应的单词索引，形状为(batch_size, knowledge_len)。其中knowledge_len是背景知识的长度。
        # ‘context’: 一个张量，表示对话上下文中每个位置对应的单词索引，形状为(batch_size, context_len)。其中context_len是对话上下文的长度。
        #
        #
        # 这个方法的返回值是一个字典，包含以下键值对：        #
        # ‘b_enc_output’: 一个张量，表示背景知识经过编码后的输出，形状为(batch_size, knowledge_len, hidden_size)。其中hidden_size是隐藏状态的维度。
        # ‘b_state’: 一个张量，表示背景知识经过编码后的最终状态，形状为(batch_size, 1, hidden_size)。
        # ‘c_enc_output’: 一个张量，表示对话上下文经过编码后的输出，形状为(batch_size, context_len, hidden_size)。
        # ‘c_state’: 一个张量，表示对话上下文经过编码后的最终状态，形状为(batch_size, 1, hidden_size)。
        # ‘segment’: 一个张量，表示知识源中最终选择的候选知识子序列向量，形状为(batch_size, 1, hidden_size)。
        # ‘p_s’: 一个张量，表示知识源中每个窗口对应的选择概率向量，形状为(batch_size, knowledge_len/window_size(4))。其中window_size是知识源中窗口的大小。
        # ‘p_g’: 一个张量，表示知识源中每个位置对应的匹配分数向量，形状为(batch_size, knowledge_len)。
        #
        # 这个方法的逻辑如下：
        # 首先调用两个编码器对象，分别对背景知识和对话上下文进行编码，得到相应的输出和状态。
        # 然后从编码器输出中取出最后一层的输出作为背景知识和对话上下文的编码输出，并从编码器状态中取出最后一层的状态作为背景知识和对话上下文的编码状态，并在第二个维度上增加一个维度。
        # 接着调用知识选择器对象，根据背景知识和对话上下文的编码输出和状态以及相应的掩码向量（用于过滤掉填充位置），得到候选知识子序列向量、选择概率向量和匹配分数向量。
        # 然后将选择概率向量和匹配分数向量在最后一个维度上拼接起来，并通过一个线性层进行非线性变换，得到候选知识子序列向量。
        # 最后将所有得到的向量作为键值对保存在一个字典中，并返回这个字典作为输出。
        b_enc_outputs, b_states = self.b_encoder(data['unstructured_knowledge'])  # [batch_size, 1, knowledge_len, hidden_size], [batch_size, 1, hidden_size]
        c_enc_outputs, c_states = self.c_encoder(data['context'])  # [batch_size, 1, context_len, hidden_size], [batch_size, 1, hidden_size]
        b_enc_output = b_enc_outputs[:, -1]  # [batch_size, knowledge_len, hidden_size]
        b_state = x2ms_adapter.tensor_api.unsqueeze(b_states[:, -1], 1)  # [batch_size, 1, hidden_size]
        c_enc_output = c_enc_outputs[:, -1]  # [batch_size, context_len, hidden_size]
        c_state = x2ms_adapter.tensor_api.unsqueeze(c_states[:, -1], 1)  # [batch_size, 1, hidden_size]
        print("b_enc_output.shape:")
        print(b_enc_output.shape)
        print("c_enc_output.shape:")
        print(c_enc_output.shape)
        print("c_state.shape:")
        print(c_state.shape)
        _, p_s, p_g = self.k_selector(b_enc_output, c_enc_output, c_state, mindspore.ops.equal(data['unstructured_knowledge'],0),
                                      mindspore.ops.equal(data['context'], 0))
        # [batch_size, 1, hidden_size], [batch_size, knowledge_len/window_size(4)], [batch_size, knowledge_len]
        s_g = x2ms_adapter.cat((x2ms_adapter.tensor_api.unsqueeze(p_s, 1), x2ms_adapter.tensor_api.unsqueeze(p_g, 1)), dim=-1)
        segment = self.segment_linear(s_g)

        return {'b_enc_output': b_enc_output, 'b_state': b_state, 'c_enc_output': c_enc_output, 'c_state': c_state,
                'segment': segment, 'p_s': p_s, 'p_g': p_g}

    # decode to end 调用
    def init_decoder_states(self, data, encode_outputs):
        return self.state_tracker.initialize(encode_outputs['segment'], encode_outputs['c_state'])

    # decode_to_end中调用
    def init_feedback_states(self, data, encode_outputs, init_decoder_states):
        return self.feedback.initialize(encode_outputs['segment'], init_decoder_states)

    # decode_to_end中调用
    def decode(self, data, previous_word, encode_outputs, previous_deocde_outputs, feedback_outputs):
        # 对编码后的输出和反馈信息进行解码，返回解码后的输出。这个方法的参数如下：
        # data: 一个字典，表示输入数据，包含以下键值对：
        # ‘unstructured_knowledge’: 一个张量，表示背景知识中每个位置对应的单词索引，形状为(batch_size, knowledge_len)。其中knowledge_len是背景知识的长度。
        # ‘context’: 一个张量，表示对话上下文中每个位置对应的单词索引，形状为(batch_size, context_len)。其中context_len是对话上下文的长度。
        # previous_word: 一个张量，表示生成文本中当前位置之前的单词索引，形状为(batch_size, 1)。
        # encode_outputs: 一个字典，表示编码后的输出，包含以下键值对：
        # ‘b_enc_output’: 一个张量，表示背景知识经过编码后的输出，形状为(batch_size, knowledge_len, hidden_size)。其中hidden_size是隐藏状态的维度。
        # ‘b_state’: 一个张量，表示背景知识经过编码后的最终状态，形状为(batch_size, 1, hidden_size)。
        # ‘c_enc_output’: 一个张量，表示对话上下文经过编码后的输出，形状为(batch_size, context_len, hidden_size)。
        # ‘c_state’: 一个张量，表示对话上下文经过编码后的最终状态，形状为(batch_size, 1, hidden_size)。
        # ‘segment’: 一个张量，表示知识源中最终选择的候选知识子序列向量，形状为(batch_size, 1, hidden_size)。
        # ‘p_s’: 一个张量，表示知识源中每个窗口对应的选择概率向量，形状为(batch_size, knowledge_len/window_size(4))。其中window_size是知识源中窗口的大小。
        # ‘p_g’: 一个张量，表示知识源中每个位置对应的匹配分数向量，形状为(batch_size, knowledge_len)。
        # previous_deocde_outputs: 一个字典，表示解码之前的输出，包含以下键值对：
        # ‘p_k’: 一个张量，表示生成文本中当前位置之前对应的复制概率分布向量，形状为(batch_size, vocab_size + dyn_vocab_size)。其中vocab_size是目标词汇表的大小，dyn_vocab_size是动态词汇表的大小。
        # ‘p_v’: 一个张量，表示生成文本中当前位置之前对应的词汇概率分布向量，形状为(batch_size, vocab_size + dyn_vocab_size)。
        # ‘state’: 一个张量，表示生成文本中当前位置之前对应的隐藏状态向量，形状为(batch_size, 1, hidden_size)。
        # feedback_outputs: 一个张量，表示反馈信息向量，形状为(batch_size, knowledge_len)。
        #
        #
        # 这个方法的返回值是一个字典，包含以下键值对：
        # ‘p_k’: 一个张量，表示生成文本中当前位置对应的复制概率分布向量，形状为(batch_size, vocab_size + dyn_vocab_size)。
        # ‘p_v’: 一个张量，表示生成文本中当前位置对应的词汇概率分布向量，形状为(batch_size, vocab_size + dyn_vocab_size)。
        # ‘state’: 一个张量，表示生成文本中当前位置对应的隐藏状态向量，形状为(batch_size, 1, hidden_size)。
        #
        # 这个方法的逻辑如下：
        # 首先将当前位置之前的单词索引通过嵌入层转换为单词向量，并在第二个维度上增加一个维度。
        # 然后将单词向量进行随机失活处理，并作为输入传入状态跟踪器对象中。
        # 状态跟踪器对象根据单词向量和之前的隐藏状态向量更新当前的隐藏状态向量，并返回更新后的向量。
        # 然后判断解码之前的输出中是否有复制概率分布向量和词汇概率分布向量，如果有，则将它们分别赋值给p_k和p_v；如果没有，则将它们分别设为None。
        # 接着调用两个生成器对象，分别对复制概率分布向量和词汇概率分布向量进行更新。
        # 生成器对象根据单词向量、隐藏状态向量、反馈信息向量、候选知识子序列向量、背景知识和对话上下文的编码输出以及相应的掩码向量（用于过滤掉填充位置），
        # 计算当前位置对应的单词和背景知识或词汇表之间的生成概率，并返回更新后的概率分布向量。
        # 最后将所有得到的向量作为键值对保存在一个字典中，并返回这个字典作为输出。
        # word_embedding = x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.nn_functional.dropout(self.embedding(previous_word), training=self.training), 1)
        word_embedding = x2ms_adapter.tensor_api.unsqueeze(self.embedding(previous_word), 1)
        states = previous_deocde_outputs['state']
        states = self.state_tracker(word_embedding, states)  # [batch_size, 1, hidden_size]

        if 'p_k' in previous_deocde_outputs:
            p_k = previous_deocde_outputs['p_k']
            p_v = previous_deocde_outputs['p_v']
        else:
            p_k = None
            p_v = None

        p_k = self.c_generator(p_k, word_embedding, states, feedback_outputs, encode_outputs['segment'],
                               encode_outputs['b_enc_output'],
                               encode_outputs['c_enc_output'], mindspore.ops.equal(data['unstructured_knowledge'],0),
                               mindspore.ops.equal(data['context'],0))
        p_v = self.v_generator(p_v, word_embedding, states, feedback_outputs, encode_outputs['segment'],
                               encode_outputs['b_enc_output'],
                               encode_outputs['c_enc_output'], mindspore.ops.equal(data['unstructured_knowledge'],0),
                               mindspore.ops.equal(data['context'],0))

        return {'p_k': p_k, 'p_v': p_v, 'state': states}

    def generate(self, data, encode_outputs, decode_outputs, softmax=True):
        p = self.mixture(decode_outputs['state'], decode_outputs['p_v'], decode_outputs['p_k'],
                         data['dyn_map'])  # [batch_size, ]
        return {'p': p}

    def decoder_to_encoder(self, data, encode_outputs, gen_response):
        # 根据编码后的输出和生成的响应，返回反馈信息向量。这个方法的参数如下：
        # data: 一个字典，表示输入数据，包含以下键值对：
        # ‘unstructured_knowledge’: 一个张量，表示背景知识中每个位置对应的单词索引，形状为(batch_size, knowledge_len)。其中knowledge_len是背景知识的长度。
        # ‘context’: 一个张量，表示对话上下文中每个位置对应的单词索引，形状为(batch_size, context_len)。其中context_len是对话上下文的长度。
        # encode_outputs: 一个字典，表示编码后的输出，包含以下键值对：
        # ‘b_enc_output’: 一个张量，表示背景知识经过编码后的输出，形状为(batch_size, knowledge_len, hidden_size)。其中hidden_size是隐藏状态的维度。
        # ‘b_state’: 一个张量，表示背景知识经过编码后的最终状态，形状为(batch_size, 1, hidden_size)。
        # ‘c_enc_output’: 一个张量，表示对话上下文经过编码后的输出，形状为(batch_size, context_len, hidden_size)。
        # ‘c_state’: 一个张量，表示对话上下文经过编码后的最终状态，形状为(batch_size, 1, hidden_size)。
        # ‘segment’: 一个张量，表示知识源中最终选择的候选知识子序列向量，形状为(batch_size, 1, hidden_size)。
        # ‘p_s’: 一个张量，表示知识源中每个窗口对应的选择概率向量，形状为(batch_size, knowledge_len/window_size(4))。其中window_size是知识源中窗口的大小。
        # ‘p_g’: 一个张量，表示知识源中每个位置对应的匹配分数向量，形状为(batch_size, knowledge_len)。
        # gen_response: 一个张量，表示生成的响应中每个位置对应的单词索引，形状为(batch_size, seq_len)。其中seq_len是序列长度。
        # 这个方法的返回值是一个张量，表示反馈信息向量，形状为(batch_size, 1, knowledge_len)。
        #
        # 这个方法的逻辑如下：
        # 首先将生成响应中每个位置对应的单词索引通过嵌入层转换为单词向量，并进行随机失活处理。
        # 然后将编码后输出中的匹配分数向量在第二个维度上增加一个维度，并初始化一个全零向量作为反馈对象的初始状态向量。
        # 接着调用反馈对象，根据单词向量和初始状态向量计算反馈信息向量，并返回最后一位状态向量。
        # 然后将反馈信息向量和匹配分数向量在第二个维度上互换，并进行批次矩阵乘法，得到响应感知权重矩阵。
        # 接着将匹配分数向量在第二个维度上互换，并进行批次矩阵乘法，并在最后一个维度上进行softmax归一化处理，得到注意力权重矩阵。
        # 然后将响应感知权重矩阵和注意力权重矩阵进行批次矩阵乘法，得到响应注意力矩阵。
        # 最后将响应注意力矩阵和匹配分数向量在第二个维度上互换，并进行批次矩阵乘法，并在第二个维度上互换回来，得到反馈信息向量。返回这个向量作为输出。
        # word_embedding = x2ms_adapter.nn_functional.dropout(self.feedback_embedding(gen_response), training=self.training)
        word_embedding = self.feedback_embedding(gen_response)
        p_g = x2ms_adapter.tensor_api.unsqueeze(encode_outputs['p_g'], 1)  # p_g [batch_size, 1, knowledge_len]
        init_feedback_states = x2ms_adapter.zeros_like(p_g)  # 第0个state为0，即h0=0    [batch_size, 1, knowledge_len]
        feedback_outputs = self.feedback(word_embedding, init_feedback_states)  # 最后一位state [batch_size, 1, hidden_size]
        response_matrix = x2ms_adapter.bmm(x2ms_adapter.tensor_api.transpose(feedback_outputs, 1, 2),
                                    p_g)  # response-aware weight matrix  [batch_size, knowledge_len, knowledge_len]
        attention_matrix = x2ms_adapter.nn_functional.softmax(x2ms_adapter.bmm(x2ms_adapter.tensor_api.transpose(p_g, 1, 2), p_g),
                                     dim=-1)  # attention weight matrix   [batch_size, knowledge_len, knowledge_len]
        response_weight = x2ms_adapter.bmm(response_matrix,
                                    attention_matrix)  # response attention matrix  [batch_size, knowledge_len, knowledge_len]
        response_attention = x2ms_adapter.tensor_api.transpose(x2ms_adapter.bmm(response_weight, x2ms_adapter.tensor_api.transpose(p_g, 1, 2)), 1,
                                                                                       2)  # [batch_size, 1, knowledge_len]
        return response_attention

    # test时调用
    def generation_to_decoder_input(self, data, indices):
        return x2ms_adapter.tensor_api.masked_fill(indices, indices >= self.vocab_size, float(self.vocab2id[UNK_WORD]))

    # test时调用
    def to_word(self, data, gen_output, k=5, sampling=False):
        gen_output = gen_output['p']
        if not sampling:
            return copy_topk(gen_output, data['vocab_map'], data['vocab_overlap'], k=k, PAD=self.vocab2id[PAD_WORD],
                             UNK=self.vocab2id[UNK_WORD], BOS=self.vocab2id[BOS_WORD])
        else:
            return randomk(gen_output[:, :self.vocab_size], k=k, PAD=self.vocab2id[PAD_WORD],
                           UNK=self.vocab2id[UNK_WORD], BOS=self.vocab2id[BOS_WORD])

    def to_sentence(self, data, batch_indices):
        # dyn_id2vocab只在predict的时候用到
        return to_copy_sentence(data, batch_indices, self.id2vocab, (data['ids'],data['dyn_id2vocab']))

    def construct(self, data):
        encode_output, all_gen_output = self.decode_to_end(data,self.vocab2id, max_target_length=10,tgt=data['response'])
        return encode_output, all_gen_output

    # def construct(self, data):
    #     return 0
    def decode_to_end(self, data, vocab2id, max_target_length=None, schedule_rate=1, softmax=False,
                      encode_outputs=None,
                      init_decoder_states=None, tgt=None, init_feedback_states=None):
        # if tgt is None:
        #     tgt = data['output']
        # 查看回复
        # print('train content: \n', tgt)
        batch_size = len(data['id'])
        if max_target_length is None:
            max_target_length = x2ms_adapter.tensor_api.x2ms_size(tgt, 1)
            print("max_target_length is:")
            print(max_target_length)
        if encode_outputs is None:
            encode_outputs = self.encode(data)
            print("encode complete")
        if init_decoder_states is None:
            init_decoder_states = self.init_decoder_states(data, encode_outputs)  # [batch_size, 1, hidden_size]
            # print('init_decoder_states: ', init_decoder_states.size())
            print("init decoder complete")
        # feedback初始化
        if init_feedback_states is None:
            feedback_outputs = self.init_feedback_states(data, encode_outputs, init_decoder_states)
            print("feedback iniet complete")
        decoder_input = new_tensor([vocab2id[BOS_WORD]] * batch_size, requires_grad=False)  # 当前的前一个词

        prob = x2ms_adapter.ones((batch_size,)) * schedule_rate
        if x2ms_adapter.is_cuda_available():
            prob = prob

        all_gen_outputs = list()  # 存储pk, pv mix后的p
        all_decode_outputs = [dict({'state': init_decoder_states})]# 存储每个解码过程的pk, pv, state
        print(all_decode_outputs)
        all_feedback_states = list()  # 保存所有的feedback state
        for t in range(max_target_length):
            # 该步骤一个循环至少耗时35秒
            # max_target_length为response的size时，即256时，会耗时接近3小时
            if t != 0:
                print("start output the round is")
                print(t)
                all_decode_inputs = tgt[:, :t]  # 左闭右开，取到的是t之前所有的词
                # 待完成：model.decoder_to_encoder()、完成feedback.forward()、修改model.decode()、测试、控制feedback维度
                # feedback，输入为encoder输出（segment）和当前生成词前面的真值
                feedback_outputs = self.decoder_to_encoder(data, encode_outputs, all_decode_inputs)  # all_decode_inputs用于GRU
            # else:
            #     print("start output the round is")
            #     print(t)
            #     dyn_map = data['dyn_map']
            #     feedback_outputs = mindspore.Tensor(numpy.zeros((1,self.vocab_size + dyn_map.shape[2])),dtype=mindspore.float32)
                all_feedback_states.append(feedback_outputs)
                # decoder_outputs, decoder_states, ...
                # 生成每个解码过程的pk, pv, state
                decode_outputs = self.decode(data, decoder_input, encode_outputs, all_decode_outputs[-1], feedback_outputs)

                # 生成pk, pv mix后的p
                print(decode_outputs)
                output = self.generate(data, encode_outputs, decode_outputs, softmax=softmax)
                print("generate output")
                all_decode_outputs.append(decode_outputs)
                all_gen_outputs.append(output)

            # 查看维度
            # print('*' * 20, 'Decoder模块', '*' * 20)
            # print('decode_outputs: ', decode_outputs)
            # print('output: ', output)
            # print('all_gen_outputs:', all_gen_outputs)
            # print('all_decode_outputs: ', all_decode_outputs)

            if schedule_rate >= 1:
                decoder_input = tgt[:, t]  # decoder前一个词使用的是真实回复的前一个词
            # elif schedule_rate <= 0:
            #     probs, ids = model.to_word(data, output, 1)
            #     decoder_input = model.generation_to_decoder_input(data, ids[:, 0])
            # else:
            #     probs, ids = model.to_word(data, output, 1)
            #     indices = model.generation_to_decoder_input(data, ids[:, 0])
            #
            #     draws = x2ms_adapter.tensor_api.long(x2ms_adapter.bernoulli(prob))
            #     decoder_input = tgt[:, t] * draws + indices * (1 - draws)

        # all_gen_outputs = torch.cat(all_gen_outputs, dim=0).transpose(0, 1).contiguous()
        print("encode_output are:")
        print(encode_outputs)
        print("all_gen_output are:")
        print(all_gen_outputs)

        return encode_outputs, all_gen_outputs
