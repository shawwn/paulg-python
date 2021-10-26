from __future__ import annotations
import builtins as py
import inspect
from typing import Callable, Any, TypeVar, Union, Type

V = TypeVar('V')


def XCHECK(x, cls: Union[py.str, Type], predicate: Callable[[Any], py.bool] = None):
    if predicate is None:
        if not isinstance(cls, py.type):
            raise ValueError(f'Expected type, got {cls}')
