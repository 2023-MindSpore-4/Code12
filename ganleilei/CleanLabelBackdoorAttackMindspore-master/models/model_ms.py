import os
from typing import List, Union

# import torch
# import torch.nn as nn
# from allennlp.modules import FeedForward
# from allennlp.nn.activations import Activation
# from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
# from transformers import AutoConfig, AutoModel, AutoTokenizer, BertModel


import mindspore.nn as nn
import mindspore as ms

from mindformers import BertConfig, BertModel, BertTokenizer
from mindspore.nn import SequentialCell
from mindformers.models.base_model import BaseModel

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ms.context.set_context(device_target="GPU")

class BERT(nn.Cell):
    def __init__(self, model_path: str, mlp_layer_num: int, class_num:int=2, hidden_dim:float=1024):
        super().__init__()
        self.mlp_layer_num = mlp_layer_num
        self.config = BertConfig.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert = BertModel(self.config, self.config.is_training, self.config.use_one_hot_embeddings)
        self.hidden_size = self.config.hidden_size
        if self.mlp_layer_num > 0:
            # self.ffn = FeedForward(input_dim=self.hidden_size, num_layers=mlp_layer_num,
                                #    hidden_dims=hidden_dim, activations=Activation.by_name('elu')())
            seq_list = [nn.Dense(self.hidden_size, hidden_dim, activation='elu')]
            for _ in range(mlp_layer_num-1):
                seq_list.append(nn.Dense(hidden_dim, hidden_dim, activation='elu'))
            
            self.ffn = SequentialCell(seq_list)
            self.linear = nn.Dense(hidden_dim, class_num)

        else:
            self.linear = nn.Dense(self.hidden_size, class_num)

    def construct(self, input_ids, type_ids, attention_masks):
        sequence_output, pooled_output, embedding_tables = self.bert.construct(input_ids, type_ids, attention_masks)
        #cls_tokens = bert_output.pooler_output
        if self.mlp_layer_num > 0:
            ffn_output = self.ffn(pooled_output)
            output = self.linear(ffn_output) # batch_size, 1(4)
        else:
            output = self.linear(pooled_output)
        return output, pooled_output

    def predict(self, input):
        # with torch.no_grad():
        encode_output = self.tokenizer.encode_plus(input)
        input_ids, input_mask = ms.tensor([encode_output['input_ids']]), ms.tensor([encode_output['attention_mask']])
        output, _ = self.construct(input_ids, input_mask)

        return nn.Softmax(output, dim=-1)

    def get_semantic_feature(self, input_text: List[str]):
        tokenizer_output = self.tokenizer(input_text, return_tensors='ms', max_length=128, padding='max_length')
        input_ids = tokenizer_output['input_ids']
        attention_mask = tokenizer_output['attention_mask']
        bert_output = self.bert(input_ids, attention_mask)
        cls_output = bert_output[1]

        return cls_output 
    

if __name__ == '__main__':
    pass