from typing import Any, Optional, List, Dict, Iterable
from pathlib import Path
import itertools

from .type_define import (
    TokenIndexer, Token, Vocabulary, IndexedTokenList
    )

# from mindnlp.transforms.tokenizers import BertTokenizer
from mindformers import BertTokenizer, BertModel

_DEFAULT_VALUE = "THIS IS A REALLY UNLIKELY VALUE THAT HAS TO BE A STRING"

class SingleIdTokenIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens as single integers.

    Registered as a `TokenIndexer` with name "single_id".

    # Parameters

    namespace : `Optional[str]`, optional (default=`"tokens"`)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.  If you
        explicitly pass in `None` here, we will skip indexing and vocabulary lookups.  This means
        that the `feature_name` you use must correspond to an integer value (like `text_id`, for
        instance, which gets set by some tokenizers, such as when using byte encoding).
    lowercase_tokens : `bool`, optional (default=`False`)
        If `True`, we will call `token.lower()` before getting an index for the token from the
        vocabulary.
    start_tokens : `List[str]`, optional (default=`None`)
        These are prepended to the tokens provided to `tokens_to_indices`.
    end_tokens : `List[str]`, optional (default=`None`)
        These are appended to the tokens provided to `tokens_to_indices`.
    feature_name : `str`, optional (default=`"text"`)
        We will use the :class:`Token` attribute with this name as input.  This is potentially
        useful, e.g., for using NER tags instead of (or in addition to) surface forms as your inputs
        (passing `ent_type_` here would do that).  If you use a non-default value here, you almost
        certainly want to also change the `namespace` parameter, and you might want to give a
        `default_value`.
    default_value : `str`, optional
        When you want to use a non-default `feature_name`, you sometimes want to have a default
        value to go with it, e.g., in case you don't have an NER tag for a particular token, for
        some reason.  This value will get used if we don't find a value in `feature_name`.  If this
        is not given, we will crash if a token doesn't have a value for the given `feature_name`, so
        that you don't get weird, silent errors by default.
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        namespace: Optional[str] = "tokens",
        lowercase_tokens: bool = False,
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        feature_name: str = "text",
        default_value: str = _DEFAULT_VALUE,
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]
        self._feature_name = feature_name
        self._default_value = default_value

     
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if self.namespace is not None:
            text = self._get_feature_value(token)
            if self.lowercase_tokens:
                text = text.lower()
            counter[self.namespace][text] += 1

     
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            text = self._get_feature_value(token)
            if self.namespace is None:
                # We could have a check here that `text` is an int; not sure it's worth it.
                indices.append(text)  # type: ignore
            else:
                if self.lowercase_tokens:
                    text = text.lower()
                indices.append(vocabulary.get_token_index(text, self.namespace))

        return {"tokens": indices}

     
    def get_empty_token_list(self):
        # IndexedTokenList
        return {"tokens": []}

    def _get_feature_value(self, token: Token) -> str:
        text = getattr(token, self._feature_name)
        if text is None:
            if self._default_value is not _DEFAULT_VALUE:
                text = self._default_value
            else:
                raise ValueError(
                    f"{token} did not have attribute {self._feature_name}. If you "
                    "want to ignore this kind of error, give a default value in the "
                    "constructor of this indexer."
                )
        return text

DEFAULT_OOV_TOKEN = "[UNK]"
class PreTrainTransformerTokenIndexer:
    '''使用 Bert 的预训练词典进行分词'''

    def __init__(
            self, 
            vocab, 
            feature_name = 'text',
            extend_vocabs: List[str] = None):
        self._bert = BertTokenizer(vocab ,tokenize_chinese_chars=False)
        self._feature_name = feature_name

        self._extend_vocab:Dict[str, Dict[str, int]] = {}
        self._invert_extend_vocab:Dict[str, List] = {}
        if extend_vocabs is not None:
            pass

    def get_empty_token_list(self):
        # IndexedTokenList
        return {"input_ids": [], "token_type_ids": [], "attention_mask": []}

    def _get_feature_value(self, token: Token) -> str:
        text = getattr(token, self._feature_name)
        if text is None:
            raise ValueError(f"{token} 没有键 {self._feature_name}")
        return text
    
    def __call__(self, text, namespace=None) -> Any:
        if isinstance(text, list):
            return [self.__call__(t, namespace) for t in text]
        
        if namespace is None:
            res = self._bert(self._get_feature_value(Token(text)))
        else:
            if text not in self._extend_vocab[namespace]:
                # OOV 是最后一个词
                res = len(self._extend_vocab[namespace])
            else:
                res = self._extend_vocab[namespace][text]
        
        return res
    
    def get_special_tokens_mask(self, token_ids: List[int]):
        return self._bert.get_special_tokens_mask(token_ids, None, True)
    
    def get_str_of_idx(self, idx:int, namespace = None):
        if namespace is None:
            return self._bert.convert_ids_to_tokens(idx)
        
        assert namespace in self._invert_extend_vocab, f'错误的名称空间{namespace}'
        iev = self._invert_extend_vocab[namespace]
        assert len(iev) >= idx >= 0, f"超出范围 {idx} 词典大小 {len(iev)}"
        return DEFAULT_OOV_TOKEN if idx == len(iev) else iev[idx]
    
    def get_voc_size(self, name_space = None) -> int:
        if name_space is None:
            return len(self._bert.get_vocab())
        return len(self._extend_vocab[name_space])
    
    def get_extend_name_spaces(self) -> List[str]:
        return list(self._extend_vocab.keys())
    
    def add_token(self, namespace:str, token:str, create_default=True, repat_check = False):
        if namespace is None:
            raise ValueError(f"不能为Bert预训练词典添加新值{token}")
            
        if namespace not in self._extend_vocab:
            if create_default:
                self._new_extend_vocab(namespace)
            else:
                raise ValueError(f"不存在的字典{namespace}")
        
        vocab = self._extend_vocab[namespace]
        if token in vocab:
            if repat_check:
                raise ValueError(f"重复添加{token}")
            else:
                return
        
        vocab[token] = len(self._invert_extend_vocab[namespace])
        self._invert_extend_vocab[namespace].append(token)
    
    def load_extend_vocab(self, vocab_path):
        if isinstance(vocab_path, str):
            vocab_path = Path(vocab_path)
        
        if vocab_path.exists():
            if vocab_path.is_dir():
                for child_path in vocab_path.rglob('*.txt'):
                    self.load_extend_vocab(child_path)
            
            elif vocab_path.is_file():
                self._build_extend_vocab_from_file(vocab_path)

            else:
                raise ValueError(f"未知错误 {vocab_path}")
        else:
            raise ValueError(f"路径不存在 {vocab_path}")
    
    def _new_extend_vocab(self, name_space:str):
        if name_space in self._extend_vocab:
            raise ValueError(f"字典重复构建 {name_space}")

        self._invert_extend_vocab[name_space] = []
        self._extend_vocab[name_space] = {}
    
    def _build_extend_vocab_from_file(self, path:Path):
        name_space = path.stem
        self._new_extend_vocab(name_space)
        
        with open(str(path.resolve()), encoding='utf8') as f:
            lines = f.readlines()
            lines = [line[:-1] if line[-1] == '\n' else line for line in lines]
            self._invert_extend_vocab[name_space].extend(lines)

            for idx, line in enumerate(lines):
                self._extend_vocab[name_space].setdefault(
                    line, idx
                )
    
    def save_extend_vocab(self, output_dir):
        if isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        
        if not output_dir.exists() or output_dir.is_file():
            raise ValueError(f"提供文件夹路径错误{output_dir}")

        for name_space in self._extend_vocab:
            target_path = output_dir / (name_space + '.txt')

            with open(str(target_path.resolve()), mode='w+', encoding='utf8') as f:
                for line in self._invert_extend_vocab[name_space]:
                    f.write(line + '\n')
    
    def regist_label_to_extend_vocab(self, field):
        name_space = None
        if hasattr(field, '_label_namespace'):
            name_space = getattr(field, '_label_namespace')
        elif hasattr(field, 'label_namespace'):
            name_space = getattr(field, 'label_namespace')
        else:
            raise ValueError(f"对象没有标签名称空间 {field}")
        
        if hasattr(field, 'labels'):
            values:Iterable = getattr(field, 'labels')
            if values is None:
                return
            for val in values:
                self.add_token(
                    namespace=name_space,
                    token=val
                    )
            
        elif hasattr(field, 'label'):
            val = getattr(field, 'label')
            if val is None:
                return
            self.add_token(namespace=name_space, token= val)
        
        else:
            raise ValueError(f"对象没有标签信息 {field}")
