from .type_define import (
    Field, DataArray, Token
)

from typing import List, Iterator, Dict, Any, Mapping, Union, Set, Tuple
from collections import defaultdict
import textwrap
from copy import deepcopy
import logging

import msadapter.pytorch as torch
from msadapter.pytorch import Tensor, tensor

from .token_indexer_type import PreTrainTransformerTokenIndexer

logger = logging.getLogger(__name__)

class ConfigurationError(ValueError):
    pass

class SequenceField(Field[DataArray]):
    """
    A `SequenceField` represents a sequence of things.  This class just adds a method onto
    `Field`: :func:`sequence_length`.  It exists so that `SequenceLabelField`, `IndexField` and other
    similar `Fields` can have a single type to require, with a consistent API, whether they are
    pointing to words in a `TextField`, items in a `ListField`, or something else.
    """

    __slots__ = []  # type: ignore

    def sequence_length(self) -> int:
        """
        How many elements are there in this sequence?
        """
        raise NotImplementedError

    def empty_field(self) -> "SequenceField":
        raise NotImplementedError

class ListField(SequenceField[DataArray]):
    """
    序列化 Field 集合
    """

    __slots__ = ["field_list"]

    def __init__(self, field_list: List[Field]) -> None:
        field_class_set = {field.__class__ for field in field_list}
        assert (
            len(field_class_set) == 1
        ), "ListFields 有且只能有一个类型 " + str(field_class_set)
        # Not sure why mypy has a hard time with this type...
        self.field_list: List[Field] = field_list

    # Sequence[Field] methods
    def __iter__(self) -> Iterator[Field]:
        return iter(self.field_list)

    def __getitem__(self, idx: int) -> Field:
        return self.field_list[idx]

    def __len__(self) -> int:
        return len(self.field_list)
    
    def as_tensor(self, **kwag) -> DataArray:
        tensor_list = [field.as_tensor(**kwag) for field in self.field_list]
        return self.field_list[0].batch_tensors(tensor_list)
    
    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:
        if len(tensor_list) == 1:
            return tensor_list[0].unsqueeze(0)
        return super().batch_tensors(tensor_list)
   
    def empty_field(self):
        '''长度至少为 1'''
        return ListField([self.field_list[0].empty_field()])

    def __str__(self) -> str:
        field_class = self.field_list[0].__class__.__name__
        base_string = f"ListField of {len(self.field_list)} {field_class}s : \n"
        return " ".join([base_string] + [f"\t {field} \n" for field in self.field_list])
    
    def sequence_length(self) -> int:
        return len(self.field_list)
    
class SpanField(Field[Tensor]):
    """
    标记一个域上的跨度
    """

    __slots__ = ["span_start", "span_end", "sequence_field"]

    def __init__(self, span_start: int, span_end: int, sequence_field: SequenceField) -> None:
        self.span_start = span_start
        self.span_end = span_end
        self.sequence_field = sequence_field

        if not isinstance(span_start, int) or not isinstance(span_end, int):
            raise TypeError(
                f"SpanFields must be passed integer indices. Found span indices: "
                f"({span_start}, {span_end}) with types "
                f"({type(span_start)} {type(span_end)})"
            )
        if span_start > span_end:
            raise ValueError(
                f"span_start must be less than span_end, " f"but found ({span_start}, {span_end})."
            )

        if span_end > self.sequence_field.sequence_length() - 1:
            raise ValueError(
                f"span_end must be <= len(sequence_length) - 1, but found "
                f"{span_end} and {self.sequence_field.sequence_length() - 1} respectively."
            )
    
    def as_tensor(
            self, **kwag
        ) -> Tensor:
        return tensor([self.span_start, self.span_end], dtype=torch.int32)

    def empty_field(self):
        return SpanField(-1, -1, self.sequence_field.empty_field())

    def __str__(self) -> str:
        return f"SpanField with spans: ({self.span_start}, {self.span_end})."

    def __eq__(self, other) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return other == (self.span_start, self.span_end)
        return super().__eq__(other)

    def __len__(self):
        return 2

class MetadataField(Field[DataArray], Mapping[str, Any]):
    """
    只保存信息而不参与运算的类型，即不转换为 Tensor
    """

    __slots__ = ["metadata"]

    def __init__(self, metadata: Any) -> None:
        self.metadata = metadata

    def __getitem__(self, key: str) -> Any:
        try:
            return self.metadata[key]  # type: ignore
        except TypeError:
            raise TypeError("your metadata is not a dict")

    def __iter__(self):
        try:
            return iter(self.metadata)
        except TypeError:
            raise TypeError("your metadata is not iterable")

    def __len__(self):
        try:
            return len(self.metadata)
        except TypeError:
            raise TypeError("your metadata has no length")
    
    def as_tensor(
            self, **kwag
        ) -> Tensor:
        return self.metadata
    
    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:
        return tensor_list

    def empty_field(self) -> "MetadataField":
        return MetadataField(None)

    def __str__(self) -> str:
        return str(self.metadata)


TextFieldTensors = Dict[str, Dict[str, Tensor]]

class TextField(SequenceField):
    """
    一系列 Token
    """

    __slots__ = ["tokens"]

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
     
    # Sequence[Token] methods
    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)
    
    def as_tensor(
            self,
            pretrain_transformer_indexer: PreTrainTransformerTokenIndexer = None,
            padding_length:int = 512,
            **kwag
        ) -> Tensor:
        org_len = len(self.tokens)
        texts = [t.text for t in self.tokens]
        indexed_token_info_lists:List[Dict[str, Any]] = pretrain_transformer_indexer(texts, None)
        # 单个请求
        indexed_token_infos = defaultdict(list)
        for item in indexed_token_info_lists:
            for key, val in item.items():
                # 去除开头结尾符号 101 102
                indexed_token_infos[key].append(val[1])

        ans = {}
        for key, val in indexed_token_infos.items():
            if len(val) == 1:
                ans[key] = tensor(val[0])
            else:
                ans[key] = tensor(val)
        
        return ans
    
    def batch_tensors(self, tensor_list: List) -> Any:
        key_to_tensors: Dict[str, List[Tensor]] = defaultdict(list)
        for tensor_dict in tensor_list:
            for key, tensor in tensor_dict.items():
                key_to_tensors[key].append(tensor)
        batched_tensors = {}
        for key, tensor_list in key_to_tensors.items():
            batched_tensor = torch.stack(tensor_list)
            batched_tensors[key] = batched_tensor
        return batched_tensors

    def sequence_length(self) -> int:
        return len(self.tokens)

    def duplicate(self):
        """ 深复制 """
        return TextField(deepcopy(self.tokens))


class SequenceLabelField(Field[Tensor]):
    """ 一个在序列上的标签对象，标签长度等于序列长度 """

    __slots__ = [
        "labels",
        "sequence_field",
        "_label_namespace",
        "_skip_indexing",
    ]

    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.
    _already_warned_namespaces: Set[str] = set()

    def __init__(
        self,
        labels: Union[List[str], List[int]],
        sequence_field: SequenceField,
        label_namespace: str = "labels",
    ) -> None:
        self.labels = labels
        self.sequence_field = sequence_field
        self._label_namespace = label_namespace
        if len(labels) != sequence_field.sequence_length():
            raise ConfigurationError(
                "Label length and sequence length "
                "don't match: %d and %d" % (len(labels), sequence_field.sequence_length())
            )

        self._skip_indexing = False
        if all(isinstance(x, int) for x in labels):
            self._indexed_labels = labels
            self._skip_indexing = True

        elif not all(isinstance(x, str) for x in labels):
            raise ConfigurationError(
                "SequenceLabelFields must be passed either all "
                "strings or all ints. Found labels {} with "
                "types: {}.".format(labels, [type(x) for x in labels])
            )

    # Sequence methods
    def __iter__(self) -> Iterator[Union[str, int]]:
        return iter(self.labels)

    def __getitem__(self, idx: int) -> Union[str, int]:
        return self.labels[idx]

    def __len__(self) -> int:
        return len(self.labels)

    def as_tensor(
            self,
            pretrain_transformer_indexer: PreTrainTransformerTokenIndexer = None,
            **kwag
        ) -> Tensor:
        return tensor(pretrain_transformer_indexer(
            self.labels , self._label_namespace
        ))
     
    def empty_field(self) -> "SequenceLabelField":
        # The empty_list here is needed for mypy
        empty_list: List[str] = []
        sequence_label_field = SequenceLabelField(empty_list, self.sequence_field.empty_field())
        sequence_label_field._indexed_labels = empty_list
        return sequence_label_field

    def __str__(self) -> str:
        length = self.sequence_field.sequence_length()
        formatted_labels = "".join(
            "\t\t" + labels + "\n" for labels in textwrap.wrap(repr(self.labels), 100)
        )
        return (
            f"SequenceLabelField of length {length} with "
            f"labels:\n {formatted_labels} \t\tin namespace: '{self._label_namespace}'."
        )


class AdjacencyField(Field[Tensor]):
    """ 标记序列上的边， 可以附带一个标签 """

    __slots__ = [
        "indices",
        "labels",
        "sequence_field",
        "label_namespace",
        "padding_num",
    ]

    def __init__(
        self,
        indices: List[Tuple[int, int]],
        sequence_field: SequenceField,
        labels: List[str] = None,
        label_namespace: str = "labels",
        padding_num = -1,
    ) -> None:
        self.indices = indices
        self.labels = labels
        self.sequence_field = sequence_field
        self.label_namespace = label_namespace
        self.padding_num = padding_num

        field_length = sequence_field.sequence_length()

        if len(set(indices)) != len(indices):
            raise ConfigurationError(f"Indices must be unique, but found {indices}")

        if not all(
            0 <= index[1] < field_length and 0 <= index[0] < field_length for index in indices
        ):
            raise ConfigurationError(
                f"边标记超出范围"
                f": {indices} and {field_length}"
            )

        if labels is not None and len(indices) != len(labels):
            raise ConfigurationError(
                f"给出标签但标签和边长度不吻合: "
                f" {labels}, {indices}"
            )

    def as_tensor(
            self,
            pretrain_transformer_indexer: PreTrainTransformerTokenIndexer = None,
            num_tokens:int = None,
            **kwag
        ) -> Tensor:
        assert num_tokens is not None, '请提供 num_tokens， 即序列长度'
        adjacency_tensor = torch.ones(num_tokens, num_tokens) * self.padding_num

        if self.labels is not None:
            indexed_labels = pretrain_transformer_indexer(
                self.labels , self.label_namespace
            )
        else:
            indexed_labels = [1] * len(self.indices)

        for idx, label in zip(self.indices, indexed_labels):
            adjacency_tensor[idx] = label

        return adjacency_tensor

    def empty_field(self) -> "AdjacencyField":

        # The empty_list here is needed for mypy
        empty_list: List[Tuple[int, int]] = []
        adjacency_field = AdjacencyField( empty_list, self.sequence_field.empty_field())
        return adjacency_field

    def __str__(self) -> str:
        length = self.sequence_field.sequence_length()
        formatted_labels = "".join(
            "\t\t" + labels + "\n" for labels in textwrap.wrap(repr(self.labels), 100)
        )
        formatted_indices = "".join(
            "\t\t" + index + "\n" for index in textwrap.wrap(repr(self.indices), 100)
        )
        return (
            f"AdjacencyField of length {length}\n"
            f"\t\twith indices:\n {formatted_indices}\n"
            f"\t\tand labels:\n {formatted_labels} ."
        )

    def __len__(self):
        return len(self.sequence_field)
    

class LabelField(Field[Tensor]):
    """ 标记一个标签 """

    __slots__ = ["label", "_label_namespace", "_label_id", "_skip_indexing"]

    def __init__(
        self, label: Union[str, int], label_namespace: str = "labels", skip_indexing: bool = False
    ) -> None:
        self.label = label
        self._label_namespace = label_namespace
        self._label_id = None
        self._skip_indexing = skip_indexing

        if skip_indexing:
            if not isinstance(label, int):
                raise ConfigurationError(
                    "非索引的标签需要是数值 int 类型"
                    "Found label = {}".format(label)
                )
            self._label_id = label
        elif not isinstance(label, str):
            raise ConfigurationError(
                "需要序列化的标签应该为字符串，否则请使用 skip_indexing = True"
                "Found label: {} with type: {}.".format(label, type(label))
            )

    def as_tensor(
            self,
            pretrain_transformer_indexer: PreTrainTransformerTokenIndexer = None,
            **kwag
        ) -> Tensor:
        if self._label_id is None:
            assert pretrain_transformer_indexer is not None, f"尚未完成初始化请提供词典"
            self._label_id = pretrain_transformer_indexer(
                self.label, self._label_namespace
            )
        return tensor([self._label_id])
    
    def empty_field(self):
        return LabelField(-1, self._label_namespace, skip_indexing=True)

    def __str__(self) -> str:
        return f"LabelField with label: {self.label} in namespace: '{self._label_namespace}'.'"

    def __len__(self):
        return 1