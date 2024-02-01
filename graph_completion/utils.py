import random
from copy import deepcopy
from inspect import signature
from logging import warning
from os import urandom
from typing import Dict

import attr
import numpy as np
import torch as pt


def filter_kwargs(cls, kwargs):
    kwargs = {k: kwargs[k] for k in signature(cls).parameters.keys() if k in kwargs}
    missing = [k for k in signature(cls).parameters.keys() if k not in kwargs]
    if len(missing) > 0:
        warning(f"Didn't find {missing} when creating {cls.__name__}")
    return kwargs


def kwarg_create(cls, kwargs: Dict):
    # creates the class given the kwargs, filtering out any which aren't in class definition
    kwargs = filter_kwargs(cls, kwargs)
    return cls(**kwargs)


@attr.s
class AbstractConf:
    def to_dict(self):
        return attr.asdict(deepcopy(self), recurse=True)

    @staticmethod
    def children() -> Dict:
        return dict()

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, dict):
            d = deepcopy(d)
            children = cls.children()
            for k in d.keys():
                if k in children:
                    d[k] = children[k].from_dict(d[k]) if d[k] is not None else None
            d = filter_kwargs(cls, d)
            d = cls(**d)
        return d

    def make(self):
        cls = type(self).OPTIONS[self.name]
        return kwarg_create(cls, self.to_dict())


def reproduce(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)
    pt.cuda.manual_seed(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False


def flatten_hpars(hpars: dict) -> dict:
    stack = list(hpars.items())
    hpars_f = dict()
    while len(stack) > 0:
        key, value = stack.pop()
        if type(value) is dict:
            for key_child, value_child in value.items():
                stack.append((f"{key}.{key_child}", value_child))
        else:
            hpars_f[key] = value
    return hpars_f


def get_random_seed() -> int:
    return int.from_bytes(urandom(4), "big")
