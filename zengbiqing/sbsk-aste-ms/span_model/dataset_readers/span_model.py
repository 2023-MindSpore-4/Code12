import logging
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union
import json
import pickle as pkl
import warnings
import itertools
from tqdm import tqdm
import numpy as np
import os
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from pathlib import Path
from Hyperparameters import Hyperparameter as hp

from ms_allennlp_fix.field_type import (
    ListField, TextField, SpanField, MetadataField,
    SequenceLabelField, AdjacencyField, LabelField
    )
from ms_allennlp_fix.type_define import Field
from ms_allennlp_fix.type_define import Token
from ms_allennlp_fix.span_utils import enumerate_spans

from ms_allennlp_fix.token_indexer_type import (
    PreTrainTransformerTokenIndexer
)

from span_model.dataset_readers.document import Document, Sentence

import mindspore as ms
from ms_fix.ops import numel

import msadapter.pytorch as torch
from msadapter.pytorch import Tensor, tensor
from msadapter.pytorch.nn import functional as F
import msadapter.pytorch.nn.utils.rnn as rnn

# New
from aste.parsing import DependencyParser
from pydantic import BaseModel
from aste.data_utils import BioesTagMaker
from span_model.dataset_readers.dep_parser import DepInstanceParser

from .util import (
    tensor_tree_info, tensor_pos_tag, 
    process_relation_labels, batch_dep_type_matrix,
    pad_end
)

# TODO 必须进行裁剪，与pytorch相比读取速度差别过大（10倍
# NOTE 可能需要编写多线程

class MissingDict(dict):
    '''提供缺失默认值的字典'''

    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)
        else:
            super().__init__()
        self._missing_val = missing_val

    def __missing__(self, key):
        return self._missing_val


def format_label_fields(dep_tree: Dict[str, Any],) :

    if 'nodes' in dep_tree:
        dep_children_dict = MissingDict(
            "",(
                ((node_idx, adj_node_idx), "1")
                    for node_idx, adj_node_idxes in enumerate(dep_tree['nodes']) for adj_node_idx in adj_node_idxes
                )
            )
    else:
        dep_children_dict = MissingDict("")

    return dep_children_dict


class Stats(BaseModel):
    entity_total:int = 0
    entity_drop:int = 0
    relation_total:int = 0
    relation_drop:int = 0
    graph_total:int=0
    graph_edges:int=0
    grid_total: int = 0
    grid_paired: int = 0


class SpanModelDataException(Exception):
    pass


class SpanModelReader:
    """
    通过 read 方法读取处理过的 JSON 文件
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexer:PreTrainTransformerTokenIndexer,
                 as_tensor : bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        # New
        self.stats = Stats()
        self.is_train = False
        self.dep_parser = DependencyParser()
        self.tag_maker = BioesTagMaker()

        self._max_span_width = max_span_width
        self._token_indexer = token_indexer

        self.as_tensor = as_tensor

        # 父类信息
        self.max_instances = kwargs['max_instances'] if 'max_instances' in kwargs else None

    def read(self, file_path, pro_bar = True):
        if pro_bar:
            data_generator = tqdm(self._read(file_path), desc="reading instances")
        else:
            data_generator = self._read(file_path)

        return data_generator

    def get_dep_labels(self, data_dir,direct=False):
        dep_labels = ["self_loop"]
        dep_type_path = os.path.join(data_dir, "dep_type.json")
        with open(dep_type_path, 'r', encoding='utf-8') as f:
            dep_types = json.load(f)
            for label in dep_types:
                if direct:
                    dep_labels.append("{}_in".format(label))
                    dep_labels.append("{}_out".format(label))
                else:
                    dep_labels.append(label)
        return dep_labels

    def get_tree_labels(self, data_dir):
        tree_labels = []
        tree_type_path = os.path.join(data_dir, "tree_type.json")
        with open(tree_type_path, 'r', encoding='utf-8') as f:
            tree_types = json.load(f)
            for label in tree_types:
                tree_labels.append(label)
        return tree_labels

    def prepare_type_dict(self, data_dir):
        dep_type_list = self.get_dep_labels(data_dir)
        types_dict = {"none": 0}
        for dep_type in dep_type_list:
            types_dict[dep_type] = len(types_dict)
        return types_dict

    def prepare_tree_dict(self,data_dir):
        tree_type_list = self.get_tree_labels(data_dir)
        types_dict = {}
        for tree_type in tree_type_list:
            types_dict[tree_type] = len(types_dict)
        return types_dict

    def _read(self, file_path: str):

        file_path = str(file_path)
        set_file_path = Path(file_path)
        status = set_file_path.stem # dev /train /test
        task_name_seed = set_file_path.parent.name.split('_') 
        task_name, seed = task_name_seed[0], task_name_seed[1]
        input_file_path =  hp.data_path / task_name / (status + '.json')
        dep_file_path = hp.data_path / task_name / (status + '.txt.dep')
        tree_file_path = hp.data_path / task_name / (status + '.txt.tree')

        with open(input_file_path, "r") as f:
            lines = f.readlines()
        all_dep_info = self.load_depfile(dep_file_path)
        all_tree_info = self.load_treefile(tree_file_path)

        self.is_train = "train" in file_path  # New
        types_dict = self.prepare_type_dict(hp.data_path)
        tree_dict = self.prepare_tree_dict(hp.data_path)

        datas = []

        for i, line in enumerate(lines):
            doc_text = json.loads(line)
            dep_info = all_dep_info[i]
            tree_info = all_tree_info[i]
            dep_children_dict = format_label_fields(doc_text["dep"][0][0])

            instance = self.text_to_instance(
                doc_text, dep_children_dict,
                dep_info, types_dict,
                tree_info, tree_dict
            )
            # NOTE 可能需要修改为 iterable 返回，因为ms不支持生成器入参

            if self.as_tensor:
                instance = Field.dict_as_tensor(
                    instance, self._token_indexer
                )
            
            # encode text like tree_info
            instance["tree_info"] = tensor_tree_info(
                instance["tree_info"],
                instance["spans"],
                instance["max_span_width"],
            )

            instance["pos_tag"] = tensor_pos_tag(
                instance["metadata"].sentences[0].text,
                instance["spans"],
            )

            instance["metadata_relation_dicts"] = process_relation_labels(
                instance['metadata'].sentences, self._token_indexer
            )

            instance['dep_type_matrix'] = batch_dep_type_matrix(instance['dep_type_matrix'])

            instance = self.extract_data(instance)
            datas.append(instance)

            if len(datas) == hp.batch_size:
                data = self.batch_data(datas)
                yield data
                datas = []
            
        if len(datas) != 0:
            yield self.batch_data(datas)
        # New
        # print(dict(file_path=input_file_path, stats=self.stats))
        self.stats = Stats()

    @staticmethod
    def extract_data(data):
        extract = {
            'text' : data['text'],
            'spans' : data['spans'],
            'dep_span_children' : data['dep_span_children'],
            'ner_labels' : data['ner_labels'],
            'relation_labels' : data['relation_labels'],
            'dep_type_matrix' : data['dep_type_matrix'],
            'tree_info' : data['tree_info'],
            'pos_tag' : data['pos_tag'],
            'metadata_relation_dicts' : data['metadata_relation_dicts'],
        }

        return extract

    # TODO 思考batch 是否应该放在reader中
    @staticmethod
    def batch_data(data_list):
        data_sample = data_list[0] if len(data_list) > 0 else None
        if isinstance(data_sample, dict):
            batched = {}
            for key in data_sample:
                child_data = [d[key] for d in data_list]
                batched[key] = SpanModelReader.batch_data(child_data)

        elif isinstance(data_sample, torch.Tensor):
            if data_sample.shape[0] == 1:
                data_list = [ten.squeeze(0) for ten in data_list]

            data_list = pad_end(data_list)
            batched = rnn.pad_sequence(data_list, batch_first=True)

        elif isinstance(data_sample, list):
            if len(data_sample) == 1:
                batched = [dl[0] for dl in data_list]
            else:
                batched = [dl for dl in data_list]

        return batched
    
    def load_depfile(self, filename):
        data = []
        with open(filename, 'r') as f:
            dep_info = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    items = line.split("\t")
                    dep_info.append({
                        "governor": int(items[0]),
                        "dependent": int(items[1]),
                        "dep": items[2],
                    })
                else:
                    if len(dep_info) > 0:
                        data.append(dep_info)
                        dep_info = []
            if len(dep_info) > 0:
                data.append(dep_info)
                dep_info = []
        return data


    def load_treefile(self, filename):
        data = []
        with open(filename, 'r') as f:
            tree_info = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    items = line.split("\t")
                    tree_info.append({
                        "word": items[0],
                        "tree_list": items[1:],
                    })
                else:
                    if len(tree_info) > 0:
                        data.append(tree_info)
                        tree_info = []
            if len(tree_info) > 0:
                data.append(tree_info)
                tree_info = []
        return data

    def _too_long(self, span):
        return span[1] - span[0] + 1 > self._max_span_width

    def _process_ner(self, span_tuples:List[Tuple[int, int]], sent:Sentence):
        ner_labels = [""] * len(span_tuples)

        for span, label in sent.ner_dict.items():
            if self._too_long(span):
                continue
            # New
            self.stats.entity_total += 1
            if span not in span_tuples:
                self.stats.entity_drop += 1
                continue
            ix = span_tuples.index(span)
            ner_labels[ix] = label

        return ner_labels

    def _process_tags(self, sent) -> List[str]:
        if not sent.ner_dict:
            return []
        spans, labels = zip(*sent.ner_dict.items())
        return self.tag_maker.run(spans, labels, num_tokens=len(sent.text))

    def _process_relations(self, span_tuples:List[Tuple[int, int]], sent:Sentence):
        relations = []
        relation_indices = []

        # Loop over the gold spans. Look up their indices in the list of span tuples and store
        # values.
        for (span1, span2), label in sent.relation_dict.items():
            # If either span is beyond the max span width, skip it.
            if self._too_long(span1) or self._too_long(span2):
                continue
            # New
            self.stats.relation_total += 1
            if (span1 not in span_tuples) or (span2 not in span_tuples):
                self.stats.relation_drop += 1
                continue
            ix1 = span_tuples.index(span1)
            ix2 = span_tuples.index(span2)
            relation_indices.append((ix1, ix2))
            relations.append(label)

        return relations, relation_indices

    def _process_grid(self, sent:Sentence):
        indices = []
        for ((a_start, a_end), (b_start, b_end)), label in sent.relation_dict.items():
            for i in [a_start, a_end]:
                for j in [b_start, b_end]:
                    indices.append((i, j))
        indices = sorted(set(indices))
        assert indices
        self.stats.grid_paired += len(indices)
        self.stats.grid_total += len(sent.text) ** 2
        return indices

    def get_adj_with_value_matrix(self,max_words_num,dep_adj_matrix, dep_type_matrix,types_dict):
        final_dep_adj_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int_)
        final_dep_type_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int_)
        for pi in range(max_words_num - 1):
            for pj in range(max_words_num - 1):
                if dep_adj_matrix[pi][pj] == 0:
                    continue
                final_dep_adj_matrix[pi][pj] = dep_adj_matrix[pi][pj]
                final_dep_type_matrix[pi][pj] = types_dict[dep_type_matrix[pi][pj]]
        return final_dep_adj_matrix, final_dep_type_matrix

    def _process_sentence(
            self, 
            sent: Sentence, 
            dataset: str,
            dep_children_dict: Dict[Tuple[int, int],List[Tuple[int, int]]],
            dep_info,
            types_dict,
            tree_info,
            tree_dict
        ):
        # Get the sentence text and define the `text_field`.
        sentence_text = [self._normalize_word(word) for word in sent.text]
        text_field_list = []
        for word in sentence_text:
            text_field_list.append(Token( text= word ))
        text_field = TextField(text_field_list)

        dep_instance_parser = DepInstanceParser(basicDependencies=dep_info, tokens=sent.text)
        dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_first_order(direct=False)
        dep_adj_matrix, dep_type_matrix = self.get_adj_with_value_matrix(
            len(sent.text), dep_adj_matrix, 
            dep_type_matrix, types_dict
        )
        spans:List[SpanField] = []
        for start, end in enumerate_spans(sentence_text, max_span_width=self._max_span_width):
            spans.append(SpanField(start, end, text_field))

        n_tokens = len(sentence_text)
        candidate_indices = [(i, j) for i in range(n_tokens) for j in range(n_tokens)]
        dep_adjs = []
        dep_adjs_indices = []
        for token_pair in candidate_indices:
            dep_adj_label = dep_children_dict[token_pair]
            if dep_adj_label:
                dep_adjs_indices.append(token_pair)
                dep_adjs.append(dep_adj_label)

        span_field = ListField(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]

        # Convert data to fields.
        # NOTE: The `ner_labels` and `coref_labels` would ideally have type
        # `ListField[SequenceLabelField]`, where the sequence labels are over the `SpanField` of
        # `spans`. But calling `as_tensor_dict()` fails on this specific data type. Matt G
        # recognized that this is an AllenNLP API issue and suggested that represent these as
        # `ListField[ListField[LabelField]]` instead.
        fields = {}
        fields["text"] = text_field
        fields["spans"] = span_field
        # fields["spacy_process_matrix"] = spacy_process_matrix

        # New
        graph = self.dep_parser.run([sentence_text])[0]
        self.stats.graph_total += numel(graph.matrix)
        self.stats.graph_edges += graph.matrix.sum()
        fields["dep_graph_labels"] = AdjacencyField(
            indices=graph.indices,
            sequence_field=text_field,
            labels=None,  # Pure adjacency matrix without dep labels for now
            label_namespace=f"{dataset}__dep_graph_labels",
        )
        self._token_indexer.regist_label_to_extend_vocab(fields['dep_graph_labels'])

        if sent.ner is not None:
            ner_labels = self._process_ner(span_tuples, sent)
            fields["ner_labels"] = ListField(
                [LabelField(entry, label_namespace=f"{dataset}__ner_labels")
                 for entry in ner_labels])
            for lab in fields['ner_labels']:
                self._token_indexer.regist_label_to_extend_vocab(lab)

            fields["tag_labels"] = SequenceLabelField(
                self._process_tags(sent), text_field, label_namespace=f"{dataset}__tag_labels"
            )
            self._token_indexer.regist_label_to_extend_vocab(fields['tag_labels'])

        if sent.relations is not None:
            relation_labels, relation_indices = self._process_relations(span_tuples, sent)
            fields["relation_labels"] = AdjacencyField(
                indices=relation_indices, sequence_field=span_field, labels=relation_labels,
                label_namespace=f"{dataset}__relation_labels")
            self._token_indexer.regist_label_to_extend_vocab(fields['relation_labels'])           

            fields["grid_labels"] = AdjacencyField(
                indices=self._process_grid(sent), sequence_field=text_field, labels=None,
                label_namespace=f"{dataset}__grid_labels"
            )
            self._token_indexer.regist_label_to_extend_vocab(fields['grid_labels'])           

        # Syntax
        dep_span_children_field = AdjacencyField(
            indices=dep_adjs_indices, sequence_field=text_field, labels=dep_adjs,
            label_namespace="dep_adj_labels")
        self._token_indexer.regist_label_to_extend_vocab(dep_span_children_field)
        
        fields["dep_span_children"] = dep_span_children_field
        dep_type_matrix_tensor = Tensor(dep_type_matrix , dtype=torch.int32) 
        dep_type_matrix_tensor = dep_type_matrix_tensor.unsqueeze(0).to(dtype=torch.int64)
        
        fields["dep_type_matrix"] = MetadataField(dep_type_matrix_tensor)
        fields["tree_info"] = MetadataField(tree_info)
        fields["tree_dict"] = MetadataField(tree_dict)
        fields["max_span_width"] = MetadataField(self._max_span_width)

        return fields

    def _process_sentence_fields(
            self, 
            doc: Document, 
            dep_children_dict,
            dep_info,
            types_dict,
            tree_info,
            tree_dict
        ):
        # Process each sentence.

        sentence_fields = [
            self._process_sentence(
                sent, doc.dataset, dep_children_dict, dep_info,
                types_dict,tree_info,tree_dict
            ) for sent in doc.sentences
        ]
        # Make sure that all sentences have the same set of keys.
        first_keys = set(sentence_fields[0].keys())
        for entry in sentence_fields:
            if set(entry.keys()) != first_keys:
                raise SpanModelDataException(
                    f"Keys do not match across sentences for document {doc.doc_key}."
                )

        # For each field, store the data from all sentences together in a ListField.
        fields = {}
        keys = sentence_fields[0].keys()
        for key in keys:
            this_field = ListField([sent[key] for sent in sentence_fields])
            fields[key] = this_field

        return fields


    def text_to_instance(
            self, doc_text: Dict[str, Any],
            dep_children_dict, dep_info, 
            types_dict, tree_info, tree_dict):
        """
        Convert a Document object into an instance.
        """
        doc = Document.from_json(doc_text)

        # Make sure there are no single-token sentences; these break things.
        sent_lengths = [len(x) for x in doc.sentences]
        if min(sent_lengths) < 2:
            msg = (f"Document {doc.doc_key} 存在较短的句子，可能会影响建模")
            warnings.warn(msg)

        fields = self._process_sentence_fields(
            doc, dep_children_dict, 
            dep_info, types_dict, 
            tree_info, tree_dict
        )

        doc.dep_children_dict = dep_children_dict
        fields["metadata"] = MetadataField(doc)

        return fields

    def _instances_from_cache_file(self, cache_filename):
        with open(cache_filename, "rb") as f:
            for entry in pkl.load(f):
                yield entry

    def _instances_to_cache_file(self, cache_filename, instances):
        with open(cache_filename, "wb") as f:
            pkl.dump(instances, f, protocol=pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

def get_span_model_reader():
    # import mindspore.dataset.text as text
    # vocabulary =  text.Vocab.from_file(str(hp.vocabulary_path))  # 预训练字典内容
    pretrainTransofrmer = PreTrainTransformerTokenIndexer(str(hp.vocabulary_path))

    smr = SpanModelReader(
        hp.params['dataset_reader']['max_span_width'],
        pretrainTransofrmer
    )
    return smr