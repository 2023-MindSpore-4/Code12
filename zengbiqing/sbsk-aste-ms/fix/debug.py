from typing import (
    Dict, Callable, Any
)
from inspect import currentframe, stack, getmodule
import re

def fcall(func, *items, **kwag):
    print(f"at: {getmodule(stack()[2][0])} {currentframe().f_back.f_back.f_lineno}")
    for idx, item in enumerate(items):
        print(f'--{idx}------>')
        func(item)
    
    for key, val in kwag.items():
        print(f'--{key}------>')
        func(val)

def fprint(*items, **kwag):
    fcall(print, *items, **kwag)

def fshape(*items, **kwag):
    fcall(lambda x:print(x.shape), *items, **kwag)

def show_dict_with_spice(item:dict, regix_pair:Dict[Any, Callable], prefix=''):
    def match(regix, target_name, target_val):
        if isinstance(regix, str):
            return re.match(regix, target_name)
        
        return isinstance(target_val, regix)

    for key, val in item.items():
        print(f'{prefix}{key}')
        
        for regix, func in regix_pair.items():
            if match(regix, key, val):
                print(f'{prefix}{func(val)}')
                break
        else:
            if isinstance(val, dict):
                show_dict_with_spice(val, regix_pair, '\t' + prefix)
            else:
                print(f"{prefix}{val}")

def show_dict_tensor_just_shape(item):
    import msadapter.pytorch as torch
    show_dict_with_spice(item, {torch.Tensor : lambda x:x.shape})