# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from transformers import BertTokenizer, AutoTokenizer
import os, codecs
# import nltk
import re
from collections import Counter
# from nltk.corpus import stopwords
# from keras.preprocessing.text import Tokenizer
import pickle as pickle
from collections import Iterable
import mindspore as ms
from mindformers import BertConfig, BertModel, BertTokenizer
from mindspore import Tensor
import numpy as np

class Iterable:
    def __init__(self, data):
        self.texts = []
        self.labels = []
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')

        for text, label in data:
            self.texts.append(text)
            self.labels.append(label)

        tokenize_out = self.tokenizer(self.texts, max_length=128, padding='max_length', return_tensors='ms')
        self.input_ids = tokenize_out['input_ids'].numpy()
        self.token_type_ids = tokenize_out['token_type_ids'].numpy()
        self.attention_mask = tokenize_out['attention_mask'].numpy()
        self.labels = np.array(self.labels, dtype=np.int32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.token_type_ids[idx], self.attention_mask[idx], self.labels[idx])


if __name__ == '__main__':
    '''
    def read_data(file_path):
        import pandas as pd
        data = pd.read_csv(file_path, sep='\t').values.tolist()
        sentences = [item[0] for item in data]
        labels = [int(item[1]) for item in data]
        processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
        return processed_data

    target_set = read_data('../data/processed_data/sst-2/train.tsv')
    '''