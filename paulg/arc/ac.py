import builtins as py
from .. import scheme


class ArcException(Exception):
    pass


def unreachable():
    assert False, "Unreachable code"


def err(msg, *args, error=ArcException):
    if py.len(args) > 0:
        msg = msg + ': ' + repr(args)[1:-1]
    raise error(msg)


def exint_p(x):
    return scheme.number_p(x) and scheme.eqv_p(x, py.int(x))


def exn_p(x):
    return isinstance(x, py.Exception)


def ar_type(x):
    if scheme.boolean_p(x):
        return 'sym'
    if scheme.symbol_p(x):
        return 'sym'
    if scheme.null_p(x):
        return 'sym'
    if scheme.procedure_p(x):
        return 'fn'
    if scheme.string_p(x):
        return 'str'
    if exint_p(x):
        return 'int'
    if scheme.number_p(x):
        return 'num'
    if scheme.hash_table_p(x):
        return 'table'
    if exn_p(x):
        return 'exception'
    err("Type: unknown type", x)
    unreachable()
