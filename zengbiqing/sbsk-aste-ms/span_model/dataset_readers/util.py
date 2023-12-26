from typing import (
    List, Dict, Tuple, Any
)
import msadapter.pytorch as torch
import msadapter.pytorch.nn.functional as F
from msadapter.pytorch import Tensor
import numpy as np

import nltk

def fields_to_batches(d, keys_to_ignore=[]):
    """
    The input is a dict whose items are batched tensors. The output is a list of dictionaries - one
    per entry in the batch - with the slices of the tensors for that entry. Here's an example.
    Input:
    d = {"a": [[1, 2], [3,4]], "b": [1, 2]}
    Output:
    res = [{"a": [1, 2], "b": 1}, {"a": [3, 4], "b": 2}].
    """
    keys = [key for key in d.keys() if key not in keys_to_ignore]

    # Make sure all input dicts have same length. If they don't, there's a problem.
    lengths = {k: len(d[k]) for k in keys}
    if len(set(lengths.values())) != 1:
        msg = f"fields have different lengths: {lengths}."
        # If there's a doc key, add it to specify where the error is.
        if "doc_key" in d:
            msg = f"For document {d['doc_key']}, " + msg
        raise ValueError(msg)

    length = list(lengths.values())[0]
    res = [{k: d[k][i] for k in keys} for i in range(length)]
    return res


def batches_to_fields(batches):
    """
    The inverse of `fields_to_batches`.
    """
    # Make sure all the keys match.
    first_keys = batches[0].keys()
    for entry in batches[1:]:
        if set(entry.keys()) != set(first_keys):
            raise ValueError("Keys to not match on all entries.")

    res = {k: [] for k in first_keys}
    for batch in batches:
        for k, v in batch.items():
            res[k].append(v)

    return res

TREE_TAG_LIST = ["RBS", "SYM", "RP", "-LRB-", "VBP", "NNS", "RRC", "PP", "NP-TMP", "EX", "NX", "SQ", "POS",
                "DT", "NNP", "WDT", "FW", "LST", "#", ",", "QP", "ADVP", "WP", "VBD", "NNPS", "LS", "SBAR",
                "PDT", "RBR", "UH", ".", "VBN", "TO", "X", "WHPP", "SINV", "VBZ", "SBARQ", "RB", "INTJ", "UCP",
                "JJR", "IN", "CC", ":", "JJ", "CD", "''", "$", "PRP$", "NN", "WHADJP", "VB", "MD", "NP", "PRT",
                "CONJP", "WRB", "VBG", "FRAG", "-RRB-", "ADJP", "WHNP", "WHADVP", "``", "PRN", "JJS", "PRP",
                "VP"]
TREE_TAG_DICT = dict(zip(TREE_TAG_LIST, list(range(1, len(TREE_TAG_LIST) + 1))))
def tensor_tree_info(tree_info:List[str], spans:Tensor, max_span_width:Tensor):
    '''
    根据 span 的范围选择 tree_info 并把标签进行索引

    tree_info: [1, seq_len, path_dep]

    spans: [1, span_num, 2]

    max_span_width: [1]

    return: [span_num, max_span_width, max_path_dep]
    '''
    max_span_width = max_span_width[0]
    tree_info = tree_info[0]
    total_tree_span_info = []
    for i in range(spans.shape[1]):
        span = spans[0][i]
        cur_info = tree_info[span[0]:span[1] + 1]
        tree_span_info = [info["tree_list"] for info in cur_info]
        tree_span_num_info = []
        for word_span in tree_span_info:
            cur_tree_span_num_info = []
            for word_info in word_span:
                cur_tree_span_num_info.append(TREE_TAG_DICT[word_info])
            tree_span_num_info.append(cur_tree_span_num_info)
        total_tree_span_info.append(tree_span_num_info)

    # PAD
    max_len_list = []
    for span in total_tree_span_info:
        len_list = [len(word) for word in span]
        max_len_list.append(max(len_list))
    word_max_len = max(max_len_list)
    for i in range(0, len(total_tree_span_info)):
        for j in range(0, len(total_tree_span_info[i])):
            total_tree_span_info[i][j].extend([0] * (word_max_len - len(total_tree_span_info[i][j])))
            total_tree_span_info[i].extend([[0] * word_max_len] * (max_span_width - len(total_tree_span_info[i])))
    tree_array = np.array(total_tree_span_info)
    cur_tag_tensor = torch.LongTensor(tree_array)
    
    return cur_tag_tensor   # [span_num, max_span_width, max_path_dep]

POS_TAG_DICT = {
    'RB':1,'RBR':1,'RBS':1,'VB':2,'VBD':2,'VBG':2,'VBN':2,'VBP':2,'VBZ':2,'JJ':3,'JJR':3,'JJS':3,
    'NN':4,'NNS':4,'NNP':4,'NNPS':4,'IN':5,'DT':6,'CC':7, 'CD':8,'RP':9
}
def tensor_pos_tag(text, spans:Tensor):
    '''
    生成句子的 POS 信息，并根据 spans 生成 id列表

    text: [seq_len]

    spans:[1, span_num, 2]

    return: [span_num, max_span_width]
    '''
    text_tagged = nltk.pos_tag(text)
    tag_list = [t[1] for t in text_tagged]
    tag_list = [POS_TAG_DICT[tag] if tag in POS_TAG_DICT.keys() else 10 for tag in tag_list]
    total_cur_tag_list: List[List] = []
    for i in range(spans.shape[1]):
        span = spans[0][i]
        id_list = tag_list[span[0]:span[1]+1]
        total_cur_tag_list.append(id_list)

    len_list = [len(t) for t in total_cur_tag_list]
    max_len = max(len_list)

    # PAD
    for g in range(len(total_cur_tag_list)):
        total_cur_tag_list[g].extend([0] * (max_len - len(total_cur_tag_list[g])))

    cur_tag_nd = np.array(total_cur_tag_list)
    cur_tag_tensor = torch.LongTensor(cur_tag_nd)

    return cur_tag_tensor

def pad_end(datas:List[Tensor], max_len = None, pad_val = 0):
    '''pad end dim to same'''
    if max_len is None:
        max_len = max([d.shape[-1] for d in datas])
    
    result = []
    for d in datas:
        pd = F.pad(d, (0, max_len - d.shape[-1]), value=pad_val)
        result.append(pd)
    
    return result

def process_relation_labels(sentences, token_indexer, dataset = None):
    '''将 relation label 转换为 idx'''
    if isinstance(sentences, list):
        return [process_relation_labels(s, token_indexer) for s in sentences]
    
    relation_active_namespace = f"{dataset}__relation_labels"
    sentence = sentences
    span_token:Dict[Tuple, str] = sentence.relation_dict
    metadata_relation_dict = {}
    for span, label in span_token.items():
        metadata_relation_dict[span] = token_indexer(
            label, relation_active_namespace
        )
    
    return [metadata_relation_dict]

def batch_dep_type_matrix(dep_type_matrixs:List[Tensor], max_size = None):
    '''将 dep_type_matrixs 填充至等长后堆叠'''
    # max_size
    if max_size is None:
        max_size = max([dep_type_matrix.shape[1] for dep_type_matrix in dep_type_matrixs])
            
    # pad
    datas = []
    for dep_type_matrix in dep_type_matrixs:
        pad_size = max_size - dep_type_matrix.shape[1]
        datas.append(torch.ops.pad(dep_type_matrix, [0, pad_size, 0, pad_size], value=0.))

    return torch.cat(datas, 0)