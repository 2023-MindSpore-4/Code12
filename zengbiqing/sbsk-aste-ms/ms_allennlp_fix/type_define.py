from copy import deepcopy
from dataclasses import dataclass
from collections import (
    OrderedDict
)
from typing import (
    Optional, Dict, MutableMapping,
    Mapping, TypeVar, Generic, List, 
    Any, Iterable, Set, Tuple, Union,
    Callable 
    )
import re, json, zlib, copy
import logging
from collections import (
    defaultdict
)
from os import PathLike
from tqdm import tqdm

class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        # TODO(brendanr): Is there some reason why we need repr here? It
        # produces horrible output for simple multi-line error messages.
        return self.message


from .util import (
    pad_sequence_to_length, namespace_match
)

import mindspore as ms

import msadapter.pytorch as torch
from msadapter.pytorch import (Tensor, Parameter, tensor)
from msadapter.pytorch.nn import Module

from mindspore.common.initializer import (
    XavierNormal, initializer
)

logger = logging.getLogger(__name__)

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = "non_padded_namespaces.txt"
_NEW_LINE_REGEX = re.compile(r"\n|\r\n")


DataArray = TypeVar(
    "DataArray", ms.Tensor, Dict[str, ms.Tensor], Dict[str, Dict[str, ms.Tensor]]
)
TextFieldTensors = Dict[str, Dict[str, Tensor]]

@dataclass(init=False, repr=False)
class Token:
    text: Optional[str]
    idx: Optional[int]
    idx_end: Optional[int]
    lemma_: Optional[str]
    pos_: Optional[str]
    tag_: Optional[str]
    dep_: Optional[str]
    ent_type_: Optional[str]
    text_id: Optional[int]
    type_id: Optional[int]

    def __init__(
        self,
        text: str = None,
        idx: int = None,
        idx_end: int = None,
        lemma_: str = None,
        pos_: str = None,
        tag_: str = None,
        dep_: str = None,
        ent_type_: str = None,
        text_id: int = None,
        type_id: int = None,
    ) -> None:
        assert text is None or isinstance(
            text, str
        )  # Some very hard to debug errors happen when this is not true.
        self.text = text
        self.idx = idx
        self.idx_end = idx_end
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.tag_ = tag_
        self.dep_ = dep_
        self.ent_type_ = ent_type_
        self.text_id = text_id
        self.type_id = type_id

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def ensure_text(self) -> str:
        """
        Return the `text` field, raising an exception if it's `None`.
        """
        if self.text is None:
            raise ValueError("Unexpected null text for token")
        else:
            return self.text

class _NamespaceDependentDefaultDict(defaultdict):
    """
    This is a [defaultdict]
    (https://docs.python.org/2/library/collections.html#collections.defaultdict) where the
    default value is dependent on the key that is passed.

    We use "namespaces" in the :class:`Vocabulary` object to keep track of several different
    mappings from strings to integers, so that we have a consistent API for mapping words, tags,
    labels, characters, or whatever else you want, into integers.  The issue is that some of those
    namespaces (words and characters) should have integers reserved for padding and
    out-of-vocabulary tokens, while others (labels and tags) shouldn't.  This class allows you to
    specify filters on the namespace (the key used in the `defaultdict`), and use different
    default values depending on whether the namespace passes the filter.

    To do filtering, we take a set of `non_padded_namespaces`.  This is a set of strings
    that are either matched exactly against the keys, or treated as suffixes, if the
    string starts with `*`.  In other words, if `*tags` is in `non_padded_namespaces` then
    `passage_tags`, `question_tags`, etc. (anything that ends with `tags`) will have the
    `non_padded` default value.

    # Parameters

    non_padded_namespaces : `Iterable[str]`
        A set / list / tuple of strings describing which namespaces are not padded.  If a namespace
        (key) is missing from this dictionary, we will use :func:`namespace_match` to see whether
        the namespace should be padded.  If the given namespace matches any of the strings in this
        list, we will use `non_padded_function` to initialize the value for that namespace, and
        we will use `padded_function` otherwise.
    padded_function : `Callable[[], Any]`
        A zero-argument function to call to initialize a value for a namespace that `should` be
        padded.
    non_padded_function : `Callable[[], Any]`
        A zero-argument function to call to initialize a value for a namespace that should `not` be
        padded.
    """

    def __init__(
        self,
        non_padded_namespaces: Iterable[str],
        padded_function: Callable[[], Any],
        non_padded_function: Callable[[], Any],
    ) -> None:
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super().__init__()

    def __missing__(self, key: str):
        if any(namespace_match(pattern, key) for pattern in self._non_padded_namespaces):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]):
        # add non_padded_namespaces which weren't already present
        self._non_padded_namespaces.update(non_padded_namespaces)

class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super().__init__(
            non_padded_namespaces, lambda: {padding_token: 0, oov_token: 1}, lambda: {}
        )

class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super().__init__(
            non_padded_namespaces, lambda: {0: padding_token, 1: oov_token}, lambda: {}
        )

def _read_pretrained_tokens(embeddings_file_uri: str) -> List[str]:
    # Moving this import to the top breaks everything (cycling import, I guess)
    # from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile
    EmbeddingsTextFile = None

    logger.info("Reading pretrained tokens from: %s", embeddings_file_uri)
    tokens: List[str] = []
    with EmbeddingsTextFile(embeddings_file_uri) as embeddings_file:
        for line_number, line in enumerate(tqdm(embeddings_file), start=1):
            token_end = line.find(" ")
            if token_end >= 0:
                token = line[:token_end]
                tokens.append(token)
            else:
                line_begin = line[:20] + "..." if len(line) > 20 else line
                logger.warning("Skipping line number %d: %s", line_number, line_begin)
    return tokens

class Vocabulary:
    def __init__(
        self,
        counter: Dict[str, Dict[str, int]] = None,
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> None:
        self._padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        self._oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN

        self._non_padded_namespaces = set(non_padded_namespaces)

        self._token_to_index = _TokenToIndexDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        self._index_to_token = _IndexToTokenDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )

        self._retained_counter: Optional[Dict[str, Dict[str, int]]] = None

        # Made an empty vocabulary, now extend it.
        self._extend(
            counter,
            min_count,
            max_vocab_size,
            non_padded_namespaces,
            pretrained_files,
            only_include_pretrained_words,
            tokens_to_add,
            min_pretrained_embeddings,
        )

    def get_namespaces(self) -> Set[str]:
        return set(self._index_to_token.keys())
    
    def get_vocab_size(self, namespace: str = "tokens") -> int:
        return len(self._token_to_index[namespace])
    
    def _extend(
        self,
        counter: Dict[str, Dict[str, int]] = None,
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
    ) -> None:
        """
        This method can be used for extending already generated vocabulary.  It takes same
        parameters as Vocabulary initializer. The `_token_to_index` and `_index_to_token`
        mappings of calling vocabulary will be retained.  It is an inplace operation so None will be
        returned.
        """
        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda: int_max_vocab_size)  # type: ignore
        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        min_pretrained_embeddings = min_pretrained_embeddings or {}
        non_padded_namespaces = set(non_padded_namespaces)
        counter = counter or {}
        tokens_to_add = tokens_to_add or {}

        self._retained_counter = counter
        # Make sure vocabulary extension is safe.
        current_namespaces = {*self._token_to_index}
        extension_namespaces = {*counter, *tokens_to_add}

        for namespace in current_namespaces & extension_namespaces:
            # if new namespace was already present
            # Either both should be padded or none should be.
            original_padded = not any(
                namespace_match(pattern, namespace) for pattern in self._non_padded_namespaces
            )
            extension_padded = not any(
                namespace_match(pattern, namespace) for pattern in non_padded_namespaces
            )
            if original_padded != extension_padded:
                raise ConfigurationError(
                    "Common namespace {} has conflicting ".format(namespace)
                    + "setting of padded = True/False. "
                    + "Hence extension cannot be done."
                )

        # Add new non-padded namespaces for extension
        self._token_to_index.add_non_padded_namespaces(non_padded_namespaces)
        self._index_to_token.add_non_padded_namespaces(non_padded_namespaces)
        self._non_padded_namespaces.update(non_padded_namespaces)

        for namespace in counter:
            pretrained_set: Optional[Set] = None
            if namespace in pretrained_files:
                pretrained_list = _read_pretrained_tokens(pretrained_files[namespace])
                min_embeddings = min_pretrained_embeddings.get(namespace, 0)
                if min_embeddings > 0:
                    tokens_old = tokens_to_add.get(namespace, [])
                    tokens_new = pretrained_list[:min_embeddings]
                    tokens_to_add[namespace] = tokens_old + tokens_new
                pretrained_set = set(pretrained_list)
            token_counts = list(counter[namespace].items())
            token_counts.sort(key=lambda x: x[1], reverse=True)
            max_vocab: Optional[int]
            try:
                max_vocab = max_vocab_size[namespace]
            except KeyError:
                max_vocab = None
            if max_vocab:
                token_counts = token_counts[:max_vocab]
            for token, count in token_counts:
                if pretrained_set is not None:
                    if only_include_pretrained_words:
                        if token in pretrained_set and count >= min_count.get(namespace, 1):
                            self.add_token_to_namespace(token, namespace)
                    elif token in pretrained_set or count >= min_count.get(namespace, 1):
                        self.add_token_to_namespace(token, namespace)
                elif count >= min_count.get(namespace, 1):
                    self.add_token_to_namespace(token, namespace)

        for namespace, tokens in tokens_to_add.items():
            for token in tokens:
                self.add_token_to_namespace(token, namespace)

    def add_token_to_namespace(self, token: str, namespace: str = "tokens") -> int:
        """
        Adds `token` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError(
                "Vocabulary tokens must be strings, or saving and loading will break."
                "  Got %s (with type %s)" % (repr(token), type(token))
            )
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def get_token_index(self, token: str, namespace: str = "tokens") -> int:
        try:
            return self._token_to_index[namespace][token]
        except KeyError:
            try:
                return self._token_to_index[namespace][self._oov_token]
            except KeyError:
                logger.error("Namespace: %s", namespace)
                logger.error("Token: %s", token)
                raise KeyError(
                    f"'{token}' not found in vocab namespace '{namespace}', and namespace "
                    f"does not contain the default OOV token ('{self._oov_token}')"
                )

    def get_token_from_index(self, index: int, namespace: str = "tokens") -> str:
        return self._index_to_token[namespace][index]    
    
    @classmethod
    def from_instances(
        cls,
        instances: Iterable["Instance"],
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> "Vocabulary":
        """
        Constructs a vocabulary given a collection of `Instances` and some parameters.
        We count all of the vocabulary items in the instances, then pass those counts
        and the other parameters, to :func:`__init__`.  See that method for a description
        of what the other parameters do.

        The `instances` parameter does not get an entry in a typical AllenNLP configuration file,
        but the other parameters do (if you want non-default parameters).
        """
        logger.info("Fitting token dictionary from dataset.")
        padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in tqdm(instances, desc="building vocab"):
            instance.count_vocab_items(namespace_token_counts)

        return cls(
            counter=namespace_token_counts,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings,
            padding_token=padding_token,
            oov_token=oov_token,
        )


IndexedTokenList = Dict[str, List[Any]]

class TokenIndexer:
    """
    指示一种方法将字符串转换为数字类型
    str -> int
    """

    default_implementation = "single_id"
    has_warned_for_as_padded_tensor = False

    def __init__(self, token_min_padding_length: int = 0) -> None:
        self._token_min_padding_length: int = token_min_padding_length

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        """
        The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
        data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
        token).  This method takes a token and a dictionary of counts and increments counts for
        whatever vocabulary items are present in the token.  If this is a single token ID
        representation, the vocabulary item is likely the token itself.  If this is a token
        characters representation, the vocabulary items are all of the characters in the token.
        """
        raise NotImplementedError

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        """
        Takes a list of tokens and converts them to an `IndexedTokenList`.
        This could be just an ID for each token from the vocabulary.
        Or it could split each token into characters and return one ID per character.
        Or (for instance, in the case of byte-pair encoding) there might not be a clean
        mapping from individual tokens to indices, and the `IndexedTokenList` could be a complex
        data structure.
        """
        raise NotImplementedError

    def indices_to_tokens(
        self, indexed_tokens: IndexedTokenList, vocabulary: Vocabulary
    ) -> List[Token]:
        """
        Inverse operations of tokens_to_indices. Takes an `IndexedTokenList` and converts it back
        into a list of tokens.
        """
        raise NotImplementedError

    def get_empty_token_list(self) -> IndexedTokenList:
        """
        Returns an `already indexed` version of an empty token list.  This is typically just an
        empty list for whatever keys are used in the indexer.
        """
        raise NotImplementedError

    def get_padding_lengths(self, indexed_tokens: IndexedTokenList) -> Dict[str, int]:
        """
        This method returns a padding dictionary for the given `indexed_tokens` specifying all
        lengths that need padding.  If all you have is a list of single ID tokens, this is just the
        length of the list, and that's what the default implementation will give you.  If you have
        something more complicated, like a list of character ids for token, you'll need to override
        this.
        """
        padding_lengths = {}
        for key, token_list in indexed_tokens.items():
            padding_lengths[key] = max(len(token_list), self._token_min_padding_length)
        return padding_lengths

    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, Tensor]:
        """
        This method pads a list of tokens given the input padding lengths (which could actually
        truncate things, depending on settings) and returns that padded list of input tokens as a
        `Dict[str, Tensor]`.  This is a dictionary because there should be one key per
        argument that the `TokenEmbedder` corresponding to this class expects in its `forward()`
        method (where the argument name in the `TokenEmbedder` needs to make the key in this
        dictionary).

        The base class implements the case when all you want to do is create a padded `LongTensor`
        for every list in the `tokens` dictionary.  If your `TokenIndexer` needs more complex
        logic than that, you need to override this method.
        """
        tensor_dict = {}
        for key, val in tokens.items():
            if val and isinstance(val[0], bool):
                tensor = Tensor(
                    pad_sequence_to_length(val, padding_lengths[key], default_value=lambda: False)
                ).bool()
            else:
                tensor = Tensor(pad_sequence_to_length(val, padding_lengths[key]), dtype=ms.int64)
            tensor_dict[key] = tensor
        return tensor_dict

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

class Field(Generic[DataArray]):
    """
    字段的抽象类
    """

    __slots__ = []  # type: ignore

    def index(self, vocab: Vocabulary):
        """
        没有返回，使用 Vocabulary 修改这个字段为 id
        """
        pass

    def as_tensor(self, **kwag) -> DataArray:
        """
        根据 padding_lengths 填充获得数据 Tensor 或更复杂的结构。
        padding_lengths 和 get_padding_lengths 返回字典的键是一致的
        """
        raise NotImplementedError

    def empty_field(self) -> "Field":
        """
        So that `ListField` can pad the number of fields in a list (e.g., the number of answer
        option `TextFields`), we need a representation of an empty field of each type.  This
        returns that.  This will only ever be called when we're to the point of calling
        :func:`as_tensor`, so you don't need to worry about `get_padding_lengths`,
        `count_vocab_items`, etc., being called on this empty field.

        We make this an instance method instead of a static method so that if there is any state
        in the Field, we can copy it over (e.g., the token indexers in `TextField`).
        """
        raise NotImplementedError

    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:  # type: ignore
        """
        默认堆叠一系列 tensor。
        """
        return tensor(torch.stack(tensor_list))

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            # With the way "slots" classes work, self.__slots__ only gives the slots defined
            # by the current class, but not any of its base classes. Therefore to truly
            # check for equality we have to check through all of the slots in all of the
            # base classes as well.
            for class_ in self.__class__.mro():
                for attr in getattr(class_, "__slots__", []):
                    if getattr(self, attr) != getattr(other, attr):
                        return False
            # It's possible that a subclass was not defined as a slots class, in which
            # case we'll need to check __dict__.
            if hasattr(self, "__dict__"):
                return self.__dict__ == other.__dict__
            return True
        return NotImplemented

    def __len__(self):
        raise NotImplementedError

    def duplicate(self):
        return deepcopy(self)
    
    @staticmethod
    def dict_as_tensor(d:dict, token_indexer = None, **kwag):

        kwag.setdefault(
            'num_tokens', 512
        )

        kwag.setdefault(
            'pretrain_transformer_indexer', token_indexer
        )

        dd = {}
        for key, value in d.items():
            if hasattr(value, 'as_tensor'):
                try:
                    dd[key] = value.as_tensor(**kwag)
                except Exception as e:
                    print(f"转换错误 {key}")
                    print(value)
                    raise e
            else:
                dd[key] = value
        return dd

class Instance(Mapping[str, Field]):
    """
    一系列 Field，底层是一个字典
    """

    __slots__ = ["fields", "indexed"]

    def __init__(self, fields: MutableMapping[str, Field]) -> None:
        self.fields = fields
        self.indexed = False

    # Add methods for `Mapping`.  Note, even though the fields are
    # mutable, we don't implement `MutableMapping` because we want
    # you to use `add_field` and supply a vocabulary.
    def __getitem__(self, key: str) -> Field:
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def add_field(self, field_name: str, field: Field, vocab: Vocabulary = None) -> None:
        """
        添加一个值，并在其他值已经索引的情况下索引新值
        """
        self.fields[field_name] = field
        if self.indexed and vocab is not None:
            field.index(vocab)

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        统计所有field的OOV情况
        """
        for field in self.fields.values():
            field.count_vocab_items(counter)

    def index_fields(self, vocab: Vocabulary) -> None:
        """
        索引所有 field
        """
        if not self.indexed:
            self.indexed = True
            for field in self.fields.values():
                field.index(vocab)

    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        """
        获取所有字段的长度信息
        """
        lengths = {}
        for field_name, field in self.fields.items():
            lengths[field_name] = field.get_padding_lengths()
        return lengths

    def as_tensor_dict(
        self, padding_lengths: Dict[str, Dict[str, int]] = None
    ) -> Dict[str, DataArray]:
        """
        默认使用所有字段的 get_padding_lengths 调用 as_tensor
        或者可以传入新值
        """
        padding_lengths = padding_lengths or self.get_padding_lengths()
        tensors = {}
        for field_name, field in self.fields.items():
            tensors[field_name] = field.as_tensor(padding_lengths[field_name])
        return tensors

    def __str__(self) -> str:
        base_string = "Instance with fields:\n"
        return " ".join(
            [base_string] + [f"\t {name}: {field} \n" for name, field in self.fields.items()]
        )

    def duplicate(self) -> "Instance":
        new = Instance({k: field.duplicate() for k, field in self.fields.items()})
        new.indexed = self.indexed
        return new

class Regularizer:
    """
    对 Tensor 正则化
    """

    default_implementation = "l2"

    def __call__(self, parameter: Tensor) -> Tensor:
        raise NotImplementedError


class RegularizerApplicator:
    """
    基于正则表达式对模型进行正则化
    """

    def __init__(self, regexes: List[Tuple[str, Regularizer]] = None) -> None:
        """
        regexes 是正则表达式和正则化的二元组
        """
        self._regularizers = regexes or []

    def __call__(self, module: Module) -> Tensor:
        """
        module 应用的对象
        """
        accumulator = 0.0
        for name, parameter in module.named_parameters():
            # We first check if the parameter needs gradient updates or not
            if parameter.requires_grad:
                # For each parameter find the first matching regex.
                for regex, regularizer in self._regularizers:
                    if re.search(regex, name):
                        penalty = regularizer(parameter)
                        accumulator = accumulator + penalty
                        break
        return accumulator

class Model(Module):
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass
    def make_output_human_readable(self, output_dict: Dict[str, Tensor]):
        pass

class SpanExtractor(Module):
    """
    提取和表示句子的跨度

    SpanExtractors take a sequence tensor of shape (batch_size, timesteps, embedding_dim)
    and indices of shape (batch_size, num_spans, 2) and return a tensor of
    shape (batch_size, num_spans, ...), forming some representation of the
    spans.
    """

    def forward(
        self,
        sequence_tensor: Tensor,
        span_indices: Tensor,
        sequence_mask: Tensor = None,
        span_indices_mask: Tensor = None,
        **kwarg
    ):
        """
        Given a sequence tensor, extract spans and return representations of
        them. Span representation can be computed in many different ways,
        such as concatenation of the start and end spans, attention over the
        vectors contained inside the span, etc.

        # Parameters

        sequence_tensor : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : `torch.LongTensor`, required.
            A tensor of shape `(batch_size, num_spans, 2)`, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the `sequence_tensor`.
        sequence_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the `indices` tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.

        # Returns

        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
        """
        raise NotImplementedError

    def get_input_dim(self) -> int:
        """
        Returns the expected final dimension of the `sequence_tensor`.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the expected final dimension of the returned span representation.
        """
        raise NotImplementedError

class TextFieldEmbedder(Module):
    """
    A `TextFieldEmbedder` is a `Module` that takes as input the
    [`DataArray`](../../data/fields/text_field.md) produced by a [`TextField`](../../data/fields/text_field.md) and
    returns as output an embedded representation of the tokens in that field.

    The `DataArrays` produced by `TextFields` are _dictionaries_ with named representations, like
    "words" and "characters".  When you create a `TextField`, you pass in a dictionary of
    [`TokenIndexer`](../../data/token_indexers/token_indexer.md) objects, telling the field how exactly the
    tokens in the field should be represented.  This class changes the type signature of `Module.forward`,
    restricting `TextFieldEmbedders` to take inputs corresponding to a single `TextField`, which is
    a dictionary of tensors with the same names as were passed to the `TextField`.

    We also add a method to the basic `Module` API: `get_output_dim()`.  You might need this
    if you want to construct a `Linear` layer using the output of this embedder, for instance.
    """

    default_implementation = "basic"

    def forward(
        self, text_field_input: TextFieldTensors, num_wrapping_dims: int = 0, **kwargs
    ) -> Tensor:
        """
        # Parameters

        text_field_input : `TextFieldTensors`
            A dictionary that was the output of a call to `TextField.as_tensor`.  Each tensor in
            here is assumed to have a shape roughly similar to `(batch_size, sequence_length)`
            (perhaps with an extra trailing dimension for the characters in each token).
        num_wrapping_dims : `int`, optional (default=`0`)
            If you have a `ListField[TextField]` that created the `text_field_input`, you'll
            end up with tensors of shape `(batch_size, wrapping_dim1, wrapping_dim2, ...,
            sequence_length)`.  This parameter tells us how many wrapping dimensions there are, so
            that we can correctly `TimeDistribute` the embedding of each named representation.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the dimension of the vector representing each token in the output of this
        `TextFieldEmbedder`.  This is _not_ the shape of the returned tensor, but the last element
        of that shape.
        """
        raise NotImplementedError


class TokenEmbedder(Module):
    """
    A `TokenEmbedder` is a `Module` that takes as input a tensor with integer ids that have
    been output from a [`TokenIndexer`](/api/data/token_indexers/token_indexer.md) and outputs
    a vector per token in the input.  The input typically has shape `(batch_size, num_tokens)`
    or `(batch_size, num_tokens, num_characters)`, and the output is of shape `(batch_size, num_tokens,
    output_dim)`.  The simplest `TokenEmbedder` is just an embedding layer, but for
    character-level input, it could also be some kind of character encoder.

    We add a single method to the basic `Module` API: `get_output_dim()`.  This lets us
    more easily compute output dimensions for the
    [`TextFieldEmbedder`](/api/modules/text_field_embedders/text_field_embedder.md),
    which we might need when defining model parameters such as LSTMs or linear layers, which need
    to know their input dimension before the layers are called.
    """

    default_implementation = "embedding"

    def get_output_dim(self) -> int:
        """
        Returns the final output dimension that this `TokenEmbedder` uses to represent each
        token.  This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

def _replace_none(params: Any) -> Any:
    if params == "None":
        return None
    elif isinstance(params, dict):
        for key, value in params.items():
            params[key] = _replace_none(value)
        return params
    elif isinstance(params, list):
        return [_replace_none(value) for value in params]
    return params

def _is_dict_free(obj: Any) -> bool:
    """
    Returns False if obj is a dict, or if it's a list with an element that _has_dict.
    """
    if isinstance(obj, dict):
        return False
    elif isinstance(obj, list):
        return all(_is_dict_free(item) for item in obj)
    else:
        return True

def infer_and_cast(value: Any):
    """
    In some cases we'll be feeding params dicts to functions we don't own;
    for example, PyTorch optimizers. In that case we can't use `pop_int`
    or similar to force casts (which means you can't specify `int` parameters
    using environment variables). This function takes something that looks JSON-like
    and recursively casts things that look like (bool, int, float) to (bool, int, float).
    """

    if isinstance(value, (int, float, bool)):
        # Already one of our desired types, so leave as is.
        return value
    elif isinstance(value, list):
        # Recursively call on each list element.
        return [infer_and_cast(item) for item in value]
    elif isinstance(value, dict):
        # Recursively call on each dict value.
        return {key: infer_and_cast(item) for key, item in value.items()}
    elif isinstance(value, str):
        # If it looks like a bool, make it a bool.
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            # See if it could be an int.
            try:
                return int(value)
            except ValueError:
                pass
            # See if it could be a float.
            try:
                return float(value)
            except ValueError:
                # Just return it as a string.
                return value
    else:
        raise ValueError(f"cannot infer type of {value}")

def with_fallback(preferred: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dicts, preferring values from `preferred`.
    """

    def merge(preferred_value: Any, fallback_value: Any) -> Any:
        if isinstance(preferred_value, dict) and isinstance(fallback_value, dict):
            return with_fallback(preferred_value, fallback_value)
        elif isinstance(preferred_value, dict) and isinstance(fallback_value, list):
            # treat preferred_value as a sparse list, where each key is an index to be overridden
            merged_list = fallback_value
            for elem_key, preferred_element in preferred_value.items():
                try:
                    index = int(elem_key)
                    merged_list[index] = merge(preferred_element, fallback_value[index])
                except ValueError:
                    raise ConfigurationError(
                        "could not merge dicts - the preferred dict contains "
                        f"invalid keys (key {elem_key} is not a valid list index)"
                    )
                except IndexError:
                    raise ConfigurationError(
                        "could not merge dicts - the preferred dict contains "
                        f"invalid keys (key {index} is out of bounds)"
                    )
            return merged_list
        else:
            return copy.deepcopy(preferred_value)

    preferred_keys = set(preferred.keys())
    fallback_keys = set(fallback.keys())
    common_keys = preferred_keys & fallback_keys

    merged: Dict[str, Any] = {}

    for key in preferred_keys - fallback_keys:
        merged[key] = copy.deepcopy(preferred[key])
    for key in fallback_keys - preferred_keys:
        merged[key] = copy.deepcopy(fallback[key])

    for key in common_keys:
        preferred_value = preferred[key]
        fallback_value = fallback[key]

        merged[key] = merge(preferred_value, fallback_value)
    return merged



class Params(MutableMapping):
    """
    Represents a parameter dictionary with a history, and contains other functionality around
    parameter passing and validation for AllenNLP.

    There are currently two benefits of a `Params` object over a plain dictionary for parameter
    passing:

    1. We handle a few kinds of parameter validation, including making sure that parameters
       representing discrete choices actually have acceptable values, and making sure no extra
       parameters are passed.
    2. We log all parameter reads, including default values.  This gives a more complete
       specification of the actual parameters used than is given in a JSON file, because
       those may not specify what default values were used, whereas this will log them.

    !!! Consumption
        The convention for using a `Params` object in AllenNLP is that you will consume the parameters
        as you read them, so that there are none left when you've read everything you expect.  This
        lets us easily validate that you didn't pass in any `extra` parameters, just by making sure
        that the parameter dictionary is empty.  You should do this when you're done handling
        parameters, by calling `Params.assert_empty`.
    """

    # This allows us to check for the presence of "None" as a default argument,
    # which we require because we make a distinction between passing a value of "None"
    # and passing no value to the default parameter of "pop".
    DEFAULT = object()

    def __init__(self, params: Dict[str, Any], history: str = "") -> None:
        self.params = _replace_none(params)
        self.history = history

    def pop(self, key: str, default: Any = DEFAULT, keep_as_dict: bool = False) -> Any:

        """
        Performs the functionality associated with dict.pop(key), along with checking for
        returned dictionaries, replacing them with Param objects with an updated history
        (unless keep_as_dict is True, in which case we leave them as dictionaries).

        If `key` is not present in the dictionary, and no default was specified, we raise a
        `ConfigurationError`, instead of the typical `KeyError`.
        """
        if default is self.DEFAULT:
            try:
                value = self.params.pop(key)
            except KeyError:
                msg = f'key "{key}" is required'
                if self.history:
                    msg += f' at location "{self.history}"'
                raise ConfigurationError(msg)
        else:
            value = self.params.pop(key, default)

        if keep_as_dict or _is_dict_free(value):
            logger.info(f"{self.history}{key} = {value}")
            return value
        else:
            return self._check_is_dict(key, value)

    def pop_int(self, key: str, default: Any = DEFAULT) -> Optional[int]:
        """
        Performs a pop and coerces to an int.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return int(value)

    def pop_float(self, key: str, default: Any = DEFAULT) -> Optional[float]:
        """
        Performs a pop and coerces to a float.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return float(value)

    def pop_bool(self, key: str, default: Any = DEFAULT) -> Optional[bool]:
        """
        Performs a pop and coerces to a bool.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        elif isinstance(value, bool):
            return value
        elif value == "true":
            return True
        elif value == "false":
            return False
        else:
            raise ValueError("Cannot convert variable to bool: " + value)

     
    def get(self, key: str, default: Any = DEFAULT):
        """
        Performs the functionality associated with dict.get(key) but also checks for returned
        dicts and returns a Params object in their place with an updated history.
        """
        default = None if default is self.DEFAULT else default
        value = self.params.get(key, default)
        return self._check_is_dict(key, value)

    def pop_choice(
        self,
        key: str,
        choices: List[Any],
        default_to_first_choice: bool = False,
        allow_class_names: bool = True,
    ) -> Any:
        """
        Gets the value of `key` in the `params` dictionary, ensuring that the value is one of
        the given choices. Note that this `pops` the key from params, modifying the dictionary,
        consistent with how parameters are processed in this codebase.

        # Parameters

        key: `str`

            Key to get the value from in the param dictionary

        choices: `List[Any]`

            A list of valid options for values corresponding to `key`.  For example, if you're
            specifying the type of encoder to use for some part of your model, the choices might be
            the list of encoder classes we know about and can instantiate.  If the value we find in
            the param dictionary is not in `choices`, we raise a `ConfigurationError`, because
            the user specified an invalid value in their parameter file.

        default_to_first_choice: `bool`, optional (default = `False`)

            If this is `True`, we allow the `key` to not be present in the parameter
            dictionary.  If the key is not present, we will use the return as the value the first
            choice in the `choices` list.  If this is `False`, we raise a
            `ConfigurationError`, because specifying the `key` is required (e.g., you `have` to
            specify your model class when running an experiment, but you can feel free to use
            default settings for encoders if you want).

        allow_class_names: `bool`, optional (default = `True`)

            If this is `True`, then we allow unknown choices that look like fully-qualified class names.
            This is to allow e.g. specifying a model type as my_library.my_model.MyModel
            and importing it on the fly. Our check for "looks like" is extremely lenient
            and consists of checking that the value contains a '.'.
        """
        default = choices[0] if default_to_first_choice else self.DEFAULT
        value = self.pop(key, default)
        ok_because_class_name = allow_class_names and "." in value
        if value not in choices and not ok_because_class_name:
            key_str = self.history + key
            message = (
                f"{value} not in acceptable choices for {key_str}: {choices}. "
                "You should either use the --include-package flag to make sure the correct module "
                "is loaded, or use a fully qualified class name in your config file like "
                """{"model": "my_module.models.MyModel"} to have it imported automatically."""
            )
            raise ConfigurationError(message)
        return value

    def as_dict(self, quiet: bool = False, infer_type_and_cast: bool = False):
        """
        Sometimes we need to just represent the parameters as a dict, for instance when we pass
        them to PyTorch code.

        # Parameters

        quiet: `bool`, optional (default = `False`)

            Whether to log the parameters before returning them as a dict.

        infer_type_and_cast: `bool`, optional (default = `False`)

            If True, we infer types and cast (e.g. things that look like floats to floats).
        """
        if infer_type_and_cast:
            params_as_dict = infer_and_cast(self.params)
        else:
            params_as_dict = self.params

        if quiet:
            return params_as_dict

        def log_recursively(parameters, history):
            for key, value in parameters.items():
                if isinstance(value, dict):
                    new_local_history = history + key + "."
                    log_recursively(value, new_local_history)
                else:
                    logger.info(f"{history}{key} = {value}")

        log_recursively(self.params, self.history)
        return params_as_dict

    def as_flat_dict(self) -> Dict[str, Any]:
        """
        Returns the parameters of a flat dictionary from keys to values.
        Nested structure is collapsed with periods.
        """
        flat_params = {}

        def recurse(parameters, path):
            for key, value in parameters.items():
                newpath = path + [key]
                if isinstance(value, dict):
                    recurse(value, newpath)
                else:
                    flat_params[".".join(newpath)] = value

        recurse(self.params, [])
        return flat_params

    def duplicate(self) -> "Params":
        """
        Uses `copy.deepcopy()` to create a duplicate (but fully distinct)
        copy of these Params.
        """
        return copy.deepcopy(self)

    def assert_empty(self, class_name: str):
        """
        Raises a `ConfigurationError` if `self.params` is not empty.  We take `class_name` as
        an argument so that the error message gives some idea of where an error happened, if there
        was one.  `class_name` should be the name of the `calling` class, the one that got extra
        parameters (if there are any).
        """
        if self.params:
            raise ConfigurationError(
                "Extra parameters passed to {}: {}".format(class_name, self.params)
            )

    def __getitem__(self, key):
        if key in self.params:
            return self._check_is_dict(key, self.params[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def _check_is_dict(self, new_history, value):
        if isinstance(value, dict):
            new_history = self.history + new_history + "."
            return Params(value, history=new_history)
        if isinstance(value, list):
            value = [self._check_is_dict(f"{new_history}.{i}", v) for i, v in enumerate(value)]
        return value

    @classmethod
    def from_file(
        cls,
        params_file: Union[str, PathLike],
        params_overrides: Union[str, Dict[str, Any]] = "",
        ext_vars: dict = None,
    ) -> "Params":
        """
        Load a `Params` object from a configuration file.

        # Parameters

        params_file: `str`

            The path to the configuration file to load.

        params_overrides: `Union[str, Dict[str, Any]]`, optional (default = `""`)

            A dict of overrides that can be applied to final object.
            e.g. {"model.embedding_dim": 10}

        ext_vars: `dict`, optional

            Our config files are Jsonnet, which allows specifying external variables
            for later substitution. Typically we substitute these using environment
            variables; however, you can also specify them here, in which case they
            take priority over environment variables.
            e.g. {"HOME_DIR": "/Users/allennlp/home"}
        """
        if ext_vars is None:
            ext_vars = {}

        file_dict = json.loads(params_file)

        if isinstance(params_overrides, dict):
            params_overrides = json.dumps(params_overrides)
        # overrides_dict = parse_overrides(params_overrides)
        overrides_dict = {}
        param_dict = with_fallback(preferred=overrides_dict, fallback=file_dict)

        return cls(param_dict)

    def to_file(self, params_file: str, preference_orders: List[List[str]] = None) -> None:
        with open(params_file, "w") as handle:
            json.dump(self.as_ordered_dict(preference_orders), handle, indent=4)

    def as_ordered_dict(self, preference_orders: List[List[str]] = None) -> OrderedDict:
        """
        Returns Ordered Dict of Params from list of partial order preferences.

        # Parameters

        preference_orders: `List[List[str]]`, optional

            `preference_orders` is list of partial preference orders. ["A", "B", "C"] means
            "A" > "B" > "C". For multiple preference_orders first will be considered first.
            Keys not found, will have last but alphabetical preference. Default Preferences:
            `[["dataset_reader", "iterator", "model", "train_data_path", "validation_data_path",
            "test_data_path", "trainer", "vocabulary"], ["type"]]`
        """
        params_dict = self.as_dict(quiet=True)
        if not preference_orders:
            preference_orders = []
            preference_orders.append(
                [
                    "dataset_reader",
                    "iterator",
                    "model",
                    "train_data_path",
                    "validation_data_path",
                    "test_data_path",
                    "trainer",
                    "vocabulary",
                ]
            )
            preference_orders.append(["type"])

        def order_func(key):
            # Makes a tuple to use for ordering.  The tuple is an index into each of the `preference_orders`,
            # followed by the key itself.  This gives us integer sorting if you have a key in one of the
            # `preference_orders`, followed by alphabetical ordering if not.
            order_tuple = [
                order.index(key) if key in order else len(order) for order in preference_orders
            ]
            return order_tuple + [key]

        def order_dict(dictionary, order_func):
            # Recursively orders dictionary according to scoring order_func
            result = OrderedDict()
            for key, val in sorted(dictionary.items(), key=lambda item: order_func(item[0])):
                result[key] = order_dict(val, order_func) if isinstance(val, dict) else val
            return result

        return order_dict(params_dict, order_func)

    def get_hash(self) -> str:
        """
        Returns a hash code representing the current state of this `Params` object.  We don't
        want to implement `__hash__` because that has deeper python implications (and this is a
        mutable object), but this will give you a representation of the current state.
        We use `zlib.adler32` instead of Python's builtin `hash` because the random seed for the
        latter is reset on each new program invocation, as discussed here:
        https://stackoverflow.com/questions/27954892/deterministic-hashing-in-python-3.
        """
        dumped = json.dumps(self.params, sort_keys=True)
        hashed = zlib.adler32(dumped.encode())
        return str(hashed)

    def __str__(self) -> str:
        return f"{self.history}Params({self.params})"



class Initializer:
    '''初始化器'''
    def __call__(self, tensor: Tensor, **kwargs) -> None:
        raise NotImplementedError

class MSInitializer(Initializer):
    '''MS 的初始化器'''

    def __init__(self, type) -> None:
        self.type = type

    def __call__(self, tensor: Tensor, **kwargs) -> None:
        return Parameter(
            initializer(self.type, tensor.shape)
        )


class InitializerApplicator:
    """
    根据正则项应用初始化
    """

    def __init__(
        self, regexes: List[Tuple[str, Initializer]] = [], prevent_regexes: List[str] =None 
    ) -> None:
        self._initializers = []
        self._prevent_regex = None
        if prevent_regexes:
            self._prevent_regex = "(" + ")|(".join(prevent_regexes) + ")"
        
        # 在这里完成初始化
        for regex, initia_param in regexes:
            initia = None
            if initia_param['type'] == 'xavier_normal':
                initia = MSInitializer('xavier_normal')
            else:
                raise ValueError(f"未知的初始化方式 {initia_param['type']}")
            self._initializers.append((regex, initia))

    def __call__(self, module: Module) -> None:
        """
        Applies an initializer to all parameters in a module that match one of the regexes we were
        given in this object's constructor.  Does nothing to parameters that do not match.

        # Parameters

        module : `Module`, required.
            The Pytorch module to apply the initializers to.
        """
        logger.info("Initializing parameters")
        unused_regexes = {initializer[0] for initializer in self._initializers}
        uninitialized_parameters = set()
        # Store which initializers were applied to which parameters.
        for name, parameter in module.named_parameters():
            for initializer_regex, initializer in self._initializers:
                allow = self._prevent_regex is None or not bool(
                    re.search(self._prevent_regex, name)
                )
                if allow and re.search(initializer_regex, name):
                    logger.info("Initializing %s using %s initializer", name, initializer_regex)
                    initializer(parameter, parameter_name=name)
                    unused_regexes.discard(initializer_regex)
                    break
            else:  # no break
                uninitialized_parameters.add(name)
        for regex in unused_regexes:
            logger.warning("Did not use initialization regex that was passed: %s", regex)
        logger.info(
            "Done initializing parameters; the following parameters are using their "
            "default initialization from their code"
        )
        uninitialized_parameter_list = list(uninitialized_parameters)
        uninitialized_parameter_list.sort()
        for name in uninitialized_parameter_list:
            logger.info("   %s", name)


class Metric:
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """

    supports_distributed = False

    def __call__(
        self, predictions: Tensor, gold_labels: Tensor, mask: Optional[Tensor]
    ):
        """
        # Parameters

        predictions : `Tensor`, required.
            A tensor of predictions.
        gold_labels : `Tensor`, required.
            A tensor corresponding to some gold label to evaluate against.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def detach_tensors(*tensors: Tensor) -> Iterable[Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, Tensor) else x for x in tensors)