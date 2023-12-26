from typing import Any, List, Tuple, Optional

import spacy
from spacy.lang.en import English
from spacy.tokens import Token, Doc

import mindspore as ms
from mindspore import Tensor
from mindspore import ops

from tqdm import tqdm

from .utils import FlexiModel

class EmptyTokenizer:
    def __init__(self, vocab) -> None:
        self.vocab = vocab
    
    def __call__(self, tokens) -> Any:
        spaces = [token != '' for token in tokens]
        return Doc(
            self.vocab,
            words= tokens,
            spaces=spaces
        )
        
    def batch(self, tokens) -> List[Doc]:
        return [self(token) for token in tokens]

class SpacyParser(FlexiModel):
    model_name = "en_core_web_sm"
    nlp: Optional[English]

    def load(self):
        if self.nlp is None:
            self.nlp = spacy.load(self.model_name)
            self.nlp.tokenizer = EmptyTokenizer(self.nlp.vocab)
# TODO 在这里进行tokenizer的临时处理，因为代码确实不清楚为什么没有调用自定义的toeknizer

class PosTagger(SpacyParser):
    def run(self, sentences: List[List[str]]) -> List[List[str]]:
        self.load()
        sentences = self.nlp.tokenizer.batch(sentences)
        token: Token
        return [
            [token.pos_ for token in doc]
            for doc in tqdm(self.nlp.pipe(sentences, disable=["ner"]))
        ]


class DependencyGraph(FlexiModel):
    indices: List[Tuple[int, int]]
    labels: List[str]
    tokens: List[str]
    heads: List[str]

    @property
    def num_nodes(self) -> int:
        return len(self.tokens)

    @property
    def matrix(self) -> Tensor:
        assert len(self.indices) == len(self.labels)
        x = ops.zeros((self.num_nodes, self.num_nodes), ms.int64)
        for i, j in self.indices:
            x[i, j] = 1
        return x


class DependencyParser(SpacyParser):
    @staticmethod
    def run_doc(d: Doc) -> DependencyGraph:
        graph = DependencyGraph(
            indices=[],
            labels=[],
            tokens=[token.text for token in d],
            heads=[token.head.text for token in d],
        )
        for i, token in enumerate(d):
            j = token.head.i
            # Symmetric edges
            if i != j:
                graph.indices.extend([(i, j), (j, i)])
                graph.labels.extend([token.dep_, token.dep_])
        return graph

    def run(self, sentences: List[List[str]]) -> List[DependencyGraph]:
        self.load()
        sentences = self.nlp.tokenizer.batch(sentences)
        return [
            self.run_doc(d) 
            for d in self.nlp.pipe(sentences, disable=["ner"])
            ]