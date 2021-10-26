import inspect
import builtins as py
from ._src import runtime as _RT
from typing import Callable, Any, TypeVar, Union, Type
from functools import partial, wraps

T = TypeVar('T')
V = TypeVar('V')


class SchemeError(Exception):
    pass


class SchemeTypeError(SchemeError, TypeError):
    pass



def eq_p(a, b) -> py.bool:
    return a is b


def eqv_p(a, b) -> py.bool:
    return a == b


def null_p(x) -> py.bool:
    return x is None


def procedure_p(x) -> py.bool:
    return inspect.isfunction(x)


def string_p(x) -> py.bool:
    return eq_p(py.type(x), py.str)


def boolean_p(x) -> py.bool:
    return eq_p(py.type(x), py.bool)


def integer_p(x) -> py.bool:
    return eq_p(py.type(x), py.int)


def float_p(x) -> py.bool:
    return eq_p(py.type(x), py.float)


def number_p(x) -> py.bool:
    return integer_p(x) or float_p(x)


def symbol_p(x) -> py.bool:
    return string_p(x) and all(part.isidentifier() for part in x.split('.'))


def hash_table_p(x) -> py.bool:
    return eq_p(py.type(x), py.dict)


def TYPEP(x: py.type) -> py.bool:
    return py.isinstance(x, py.type)


def INSTANCEP(x, cls) -> py.bool:
    if procedure_p(cls):
        return cls(x)
    if not TYPEP(cls):
        raise TypeError(f'Expected type, got {cls!r}')
    return py.isinstance(x, cls)


def XCHECK(x, cls: Union[Callable[[T], py.bool], Type[T]], name=None) -> T:
    if not INSTANCEP(x, cls):
        raise TypeError(f'Expected {name or py.str(cls)}, got {x}')
    return x


def INTSAFE(x):
    try:
        return py.int(x)
    except:
        return x


def EXACTP(x):
    return integer_p(x) or eqv_p(INTSAFE(x), x)


def XINT(x) -> py.int:
    return py.int(XCHECK(x, EXACTP, 'integer'))


def XNUM(x) -> py.int:
    return py.int(XCHECK(x, number_p, 'number'))


def XBOOL(x) -> py.bool:
    return XCHECK(x, boolean_p, 'boolean')


def XSTR(x) -> py.str:
    return XCHECK(x, string_p, 'string')


def XSYM(x) -> py.str:
    return XCHECK(x, symbol_p, 'symbol')


def string_length(x):
    s = XSTR(x)
    return py.len(s)


def symbol_to_string(x):
    s: py.str = XSYM(x)
    return py.str(s)
