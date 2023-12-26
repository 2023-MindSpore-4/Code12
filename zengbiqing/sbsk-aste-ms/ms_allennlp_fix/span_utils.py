from typing import List, Callable, Tuple, TypeVar

from .type_define import Token
T = TypeVar("T", str, Token)

def enumerate_spans(
    sentence: List[T],
    offset: int = 0,
    max_span_width: int = None,
    min_span_width: int = 1,
    filter_function: Callable[[List[T]], bool] = None,
) -> List[Tuple[int, int]]:
    """
    枚举句子可行的 SPAN
    可以自定义过滤器
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end))
    return spans