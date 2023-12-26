from modules.Generations import *
from evaluate.Eval_Rouge import *
from evaluate.Eval_Bleu import *
from evaluate.Eval_F1 import *
from Utils import *
import sys
import json
import mindspore
import x2ms_adapter
import x2ms_adapter.datasets as x2ms_datasets
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_cell
import x2ms_adapter.nn_init
import x2ms_adapter.util_api as util_api
from RFMWoWDataset import collate_fn




def train_embedding(model):
    for name, param in x2ms_adapter.named_parameters(model):
        if 'embedding' in name:
            print('requires_grad', name, x2ms_adapter.tensor_api.x2ms_size(param))
            param.requires_grad = True


def init_params(model, escape=None):
    for name, param in x2ms_adapter.named_parameters(model):
        if escape is not None and escape in name:
            print('no_init', name, x2ms_adapter.tensor_api.x2ms_size(param))
            continue
        print('init', name, x2ms_adapter.tensor_api.x2ms_size(param))
        if x2ms_adapter.tensor_api.x2ms_dim(param.data) > 1:
            x2ms_adapter.nn_init.xavier_uniform_(param.data)


class DefaultTrainer(object):
    def __init__(self, model, local_rank):
        super(DefaultTrainer, self).__init__()
        self.local_rank = local_rank

        if local_rank is not None:
            x2ms_adapter.cuda_set_device(local_rank)

        if x2ms_adapter.is_cuda_available():
            self.model = model
        else:
            self.model = model
        self.eval_model = self.model

        # if x2ms_adapter.is_cuda_available() and local_rank is not None:
        #     print("GPU ", self.local_rank)
        #     self.model = x2ms_nn.DistributedDataParallel(self.model, device_ids=[local_rank],
        #                                                            output_device=local_rank,
        #                                                            find_unused_parameters=True)

    def train_batch(self, epoch, data, method, optimizer):
        print(data)
        # loss = self.model(data, method=method)
        encode_output, all_gen_output = self.model(data)
        print("encode_output are:")
        print(encode_output)
        print("all_gen_output are:")
        print(all_gen_output)
        output_list = [encode_output, all_gen_output]
        loss = self.forward(output_list,data)
        print("initial loss is:")
        print(loss)
        if isinstance(loss, tuple) or isinstance(loss, list):
            closs = [x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.mean(l)) for l in loss]
            print("closs is:")
            print(closs)
            loss = x2ms_adapter.tensor_api.mean(x2ms_adapter.cat(loss, dim=-1))
            print("loss is:")
            print(loss)
        else:
            loss = x2ms_adapter.tensor_api.mean(loss)
            closs = [x2ms_adapter.tensor_api.item(loss)]


        # util_api.clip_grad_norm(self.model.trainable_params(), 2)

        return closs

    def serialize(self, epoch, output_path):
        if self.local_rank != 0:
            return
        output_path = os.path.join(output_path, 'model/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        x2ms_adapter.save(x2ms_adapter.state_dict(self.eval_model), os.path.join(output_path, '.'.join([str(epoch), 'pkl'])))


    def train_epoch(self, method, train_dataset, train_collate_fn, batch_size, epoch, optimizer):
        # 根据输入数据和类型，进行一个训练周期，并打印损失值和时间。这个方法的参数如下：
        # method: 一个字符串，表示训练的类型。可以是’mle_train’、‘mcc_train’、'ds_train’或者它们的组合。
        # train_dataset: 一个数据集对象，表示训练数据集。
        # train_collate_fn: 一个函数对象，表示训练数据集的整理函数，用于将多个样本拼接成一个批次。
        # batch_size: 一个整数，表示每个批次的样本数。
        # epoch: 一个整数，表示当前的训练周期数。
        # optimizer: 一个优化器对象，表示用于更新模型参数的优化器。
        # 这个方法没有返回值。
        #
        # 这个方法的逻辑如下：
        # 首先将模型设置为训练模式，并判断是否有可用的CUDA设备。
        # 如果有，则使用分布式采样器对训练数据集进行采样，并创建一个数据加载器对象，用于按照批次加载训练数据。
        # 如果没有，则直接创建一个数据加载器对象，并设置随机打乱数据的选项。
        # 然后记录开始时间，并初始化一个计数器，用于记录处理过的批次数。
        # 接着遍历数据加载器中的每个批次数据，并判断是否有可用的CUDA设备。
        # 如果有，则将每个批次数据中的张量转换为CUDA张量，并保存在一个新的字典中。如果没有，则直接使用原始的批次数据。
        # 然后将计数器加一，并调用train_batch方法，根据当前的训练周期、批次数据、类型和优化器，进行一次批次训练，并返回损失值。
        # 如果当前处理过的批次数是100的倍数，则计算经过的时间，并打印出类型、周期、批次、损失和时间等信息，并刷新标准输出流。
        # 最后删除损失值变量，释放内存空间。
        x2ms_adapter.x2ms_train(self.model)
        # if x2ms_adapter.is_cuda_available():
        #     sampler = x2ms_datasets.DistributedSampler(train_dataset)
        #     train_loader = x2ms_datasets.DataLoader(train_dataset, collate_fn=train_collate_fn,
        #                                                batch_size=batch_size, sampler=sampler)
        # else:
        #     train_loader = x2ms_datasets.DataLoader(train_dataset, collate_fn=train_collate_fn,
        #                                                batch_size=batch_size, shuffle=True)
        # train_loader = collate_fn(train_dataset)
        train_loader = train_dataset
        start_time = time.time()
        count_batch = 0
        data_cuda = dict()
        for key, value in train_loader.items():
            if isinstance(value, mindspore.Tensor):
                data_cuda[key] = value
            else:
                data_cuda[key] = value
        data = data_cuda
        count_batch += 1
        data['dyn_map'] = build_map(data['dyn_map'])
        bloss = self.train_batch(epoch, data, method=method, optimizer=optimizer)

        elapsed_time = time.time() - start_time
        print('Method', method, 'Epoch', epoch, 'Batch ', count_batch, 'Loss ', bloss, 'Time ', elapsed_time)
        sys.stdout.flush()
        del bloss

        # elapsed_time = time.time() - start_time
        # print(method + ' ', epoch, 'time ', elapsed_time)
        sys.stdout.flush()

    def predict(self, method, dataset, collate_fn, batch_size, epoch, output_path, test_type):
        x2ms_adapter.x2ms_eval(self.eval_model)
        test_loader = x2ms_datasets.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=0)
        srcs = []
        systems = []
        references = []
        for k, data in enumerate(test_loader, 0):
            if x2ms_adapter.is_cuda_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, mindspore.Tensor):
                        data_cuda[key] = value
                    else:
                        data_cuda[key] = value
                data = data_cuda

            indices = self.eval_model(data, method=method)
            sents = self.eval_model.to_sentence(data, indices)

            remove_duplicate(sents)

            srcs += [' '.join(dataset.input(x2ms_adapter.tensor_api.item(id))) for id in data['id']]
            systems += [' '.join(s).replace(SEP_WORD, os.linesep).lower() for s in sents]
            if test_type == 'SR':
                for id in data['id']:
                    refs = dataset.output(x2ms_adapter.tensor_api.item(id))
                    # print('refs: ', refs)
                    refs = [' '.join(ref).lower() for ref in refs]
                    references.append(refs)

            else:
                with open("data/modified_multi_reference_test.json", 'r', encoding='utf-8') as r:
                    multi_reference_test = json.load(r)
                for e_id in data['example_id']:
                    references.append(multi_reference_test[e_id]['responses'])

        output_path = os.path.join(output_path, 'result/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        file = codecs.open(os.path.join(output_path, str(epoch) + '.txt'), "w", "utf-8")
        for i in range(len(systems)):
            # print('srcs: ', srcs[i])
            # print('system: ', systems[i])
            # print('references: ', references[i])
            file.write(srcs[i] + os.linesep + systems[i] + os.linesep + os.linesep.join(references[i]) + os.linesep + os.linesep)
        file.close()
        return systems, references

    def test(self, method, dataset, collate_fn, batch_size, epoch, output_path, test_type):
        systems, references = self.predict(method, dataset, collate_fn, batch_size, epoch, output_path, test_type)

        rouges = eval_rouge(systems, references)
        f1 = eval_f1(systems, references)
        # bleus =eval_bleu(systems, references)

        print({**rouges})
        print({**f1})
        sys.stdout.flush()
        return rouges, f1  # , bleus

    def do_train(self, outputlist, data):
        # 根据输入数据和类型，进行训练，并返回损失值。这个方法的参数如下：
        # data: 一个字典，表示输入数据，包含以下键值对：
        # ‘response’: 一个张量，表示参考响应中每个位置对应的单词索引，形状为(batch_size, seq_len)。其中seq_len是序列长度。
        # ‘dyn_response’: 一个张量，表示参考响应中每个位置对应的动态词汇表中的单词索引，形状为(batch_size, seq_len)。
        # ‘selection’: 一个张量，表示知识源中每个窗口对应的选择概率向量，形状为(batch_size, knowledge_len/window_size(4))。其中window_size是知识源中窗口的大小。
        # type: # 一个字符串，表示训练的类型。可以是’mle_train’、‘mcc_train’、'ds_train’或者它们的组合。
        #
        # 这个方法的返回值是一个列表，包含以下元素：
        # r_loss: 一个标量，表示生成响应和参考响应之间的负对数似然损失值。只有当’type’参数中包含’mle’时才返回。
        # e1_loss: 一个标量，表示生成响应中词汇表部分的熵损失值。只有当’type’参数中包含’mcc’时才返回。
        # e2_loss: 一个标量，表示生成响应中动态词汇表部分的熵损失值。只有当’type’参数中包含’mcc’时才返回。
        # k_loss: 一个标量，表示知识源中选择概率向量和真实选择概率向量之间的KL散度损失值。只有当’type’参数中包含’ds’时才返回。
        #
        # 这个方法的逻辑如下：
        # 首先调用decode_to_end函数，根据输入数据和目标词汇表，进行解码，并返回编码后的输出、解码器的初始状态、所有解码后的输出、所有生成后的输出和所有反馈信息状态。
        # 然后初始化一个空列表作为损失值的容器。
        # 如果’type’参数中包含’mle’，则从所有生成后的输出中取出每个位置对应的单词概率分布向量，并在第二个维度上拼接起来，并展平为二维张量。然后调用损失函数对象，根据单词概率分布向量、参考响应、动态参考响应和未知单词在目标词汇表中的索引，计算负对数似然损失值，并在第一个维度上增加一个维度。将这个损失值添加到损失值列表中。
        # 如果’type’参数中包含’mcc’，则从所有生成后的输出中取出每个位置对应的单词概率分布向量，并在第二个维度上拼接起来，并展平为二维张量。然后将这个张量按照目标词汇表和动态词汇表部分进行切分，并分别计算它们的熵值，并取平均值。然后将这两个熵值乘以0.1并减去1，得到两个熵损失值。将这两个损失值添加到损失值列表中。
        # 如果’type’参数中包含’ds’，则从编码后的输出中取出每个窗口对应的选择概率向量，并在第二个维度上增加一个维度。然后调用KL散度函数，根据选择概率向量和输入数据中的真实选择概率向量，在最后一个维度上进行softmax归一化处理，并计算KL散度损失值，并在第一个维度上增加一个维度。将这个损失值添加到损失值列表中。
        # 最后返回损失值列表作为输出。
        encode_output = outputlist[0]
        # init_decoder_state = outputlist[1]
        # all_decode_output = outputlist[2]
        all_gen_output = outputlist[1]
        # all_feedback_states = outputlist[4]
        loss = list()
        # if 'mle' in type:
        p = x2ms_adapter.cat([x2ms_adapter.tensor_api.unsqueeze(p['p'], 1) for p in all_gen_output], dim=1)
        p = x2ms_adapter.tensor_api.view(p, -1, x2ms_adapter.tensor_api.x2ms_size(p, -1))

        if data['dyn_response'].shape[1] != data['response'].shape[1]:
            pad_length = data['response'].shape[1] - data['dyn_response'].shape[1]
            paddings = ((0, 0), (0, pad_length))
            dyn_response = mindspore.ops.Pad(paddings)(data['dyn_response'])
        else:
            dyn_response = data['dyn_response']
        print("this is new dyn_response.shape:")
        print(dyn_response.shape)
        r_loss = x2ms_adapter.tensor_api.unsqueeze(self.model.criterion(p, data['response'], dyn_response, self.model.vocab2id[UNK_WORD], reduction='mean'), 0)
        print("r_loss is:")
        print(r_loss)
        loss += [r_loss]
        # if 'mcc' in type:
        # sub_p1 = p[:, :self.model.vocab_size]
        # row_sums1 = x2ms_adapter.x2ms_sum(sub_p1, dim=1, keepdim=True)
        # normalized_p1 = sub_p1 / row_sums1
        # e1_loss = 1 - 0.1 * x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.mean(nn.probability.distribution.Categorical(probs=normalized_p1 + self.model.eps).entropy()), 0)
        # e2_loss = 1 - 0.1 * x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.mean(nn.probability.distribution.Categorical(probs=p[:, self.model.vocab_size:] + self.model.eps).entropy()), 0)
        # print("e1_loss ,e2_loss are:")
        # print(e1_loss)
        # print(e2_loss)
        # loss += [e1_loss, e2_loss]

        # if 'ds' in type:
        k_loss = x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.nn_functional.kl_div(x2ms_adapter.tensor_api.log((x2ms_adapter.tensor_api.squeeze(encode_output['p_s'], 1) + self.model.eps)),data['selection'] + self.model.eps,reduction='batchmean'), 0)
        print("k_loss is:")
        print(k_loss)
        loss += [k_loss]

        return loss

    def forward(self, output_list, data):
        return self.do_train(output_list, data)