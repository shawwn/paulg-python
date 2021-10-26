from __future__ import annotations
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import operator
from operator import attrgetter
from contextlib import contextmanager
from collections import namedtuple
from functools import total_ordering
import gc
import itertools as it
from weakref import ref
import threading
import types
from typing import (Any, Callable, ClassVar, Dict, Generator,
                    Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple,
                    Type, Union, cast, Iterable, Hashable)

import numpy as np

# from ._src import dtypes
from ._src import config as jax_config
from ._src.config import FLAGS, config

import types

from . import errors
from ._src import config
from . import errors
from paulg.lisp._src import source_info_util

from ._src.util import (safe_zip, safe_map, partial, curry, prod, partialmethod,
                   tuple_insert, tuple_delete, as_hashable_function,
                   HashableFunction)



class _LispPrimitiveBase:
    pass


class LispPrimitive(_LispPrimitiveBase):
    name: str
    multiple_results = False  # set for multi-output primitives
    call_primitive = False    # set for call primitives processed in final style
    map_primitive = False     # set for map primitives processed in final style
    _dispatch_on_params = False  # whether to include axis names from params in dispatch

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return '{}'.format(self.name)


    def bind(self, *args, **params):
        assert (not config.jax_enable_checks or
                all(isinstance(arg, LispTracer) or valid_jaxtype(arg) for arg in args)), args
        top_trace = find_top_trace(
            args, used_axis_names(self, params) if self._dispatch_on_params else None)
        tracers = map(top_trace.full_raise, args)
        out = top_trace.process_primitive(self, tracers, params)
        return map(full_lower, out) if self.multiple_results else full_lower(out)

    def def_impl(self, impl):
        self.impl = impl
        return impl

    def def_abstract_eval(self, abstract_eval):
        self.abstract_eval = abstract_eval
        return abstract_eval

    def def_custom_bind(self, bind):
        self.bind = bind
        return bind

    def impl(self, *args, **params):
        raise NotImplementedError("Evaluation rule for '{}' not implemented"
                                  .format(self.name))

    def abstract_eval(self, *args, **params):
        raise NotImplementedError("Abstract evaluation for '{}' not implemented"
                                  .format(self.name))




# -------------------- tracing --------------------


class _LispTraceBase:
    pass

class LispTrace(_LispTraceBase):
    __slots__ = ['main', 'level', 'sublevel']

    main: 'MainTrace'
    level: int
    sublevel: 'Sublevel'

    def __init__(self, main: 'MainTrace', sublevel: 'Sublevel') -> None:
        self.main = main
        self.level = main.level
        self.sublevel = sublevel

    def full_raise(self, val) -> 'LispTracer':
        if not isinstance(val, LispTracer):
            return self.pure(val)
        val._assert_live()
        level = self.level
        sublevel = self.sublevel
        if val._trace.main is self.main:
            if val._trace.sublevel == sublevel:
                return val
            elif val._trace.sublevel < sublevel:
                return self.sublift(val)
            else:
                raise escaped_tracer_error(
                    val, f"Can't lift sublevels {val._trace.sublevel} to {sublevel}")
        elif val._trace.level < level:
            if val._trace.sublevel > sublevel:
                raise escaped_tracer_error(
                    val, f"Incompatible sublevel: {val._trace}, {(level, sublevel)}")
            return self.lift(val)
        elif val._trace.level > level:
            raise escaped_tracer_error(
                val, f"Can't lift level {val} to {self}")
        else:  # val._trace.level == self.level:
            raise escaped_tracer_error(
                val, f"Different traces at same level: {val}, {self}")

    def pure(self, val):
        raise NotImplementedError("must override")

    def lift(self, tracer):
        raise NotImplementedError("must override")

    def sublift(self, tracer):
        raise NotImplementedError("must override")

    def process_primitive(self, primitive, tracers, params):
        raise NotImplementedError("must override")

    def __repr__(self):
        return '{}(level={}/{})'.format(
            self.__class__.__name__, self.level, self.sublevel)

    def process_call(self, call_primitive, f, tracers, params):
        msg = (f"{type(self)} must override process_call to handle call-like "
               "primitives")
        raise NotImplementedError(msg)

    def process_map(self, map_primitive, f, tracers, params):
        msg = (f"{type(self)} must override process_map to handle map-like "
               "primitives")
        raise NotImplementedError(msg)

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers):
        msg = (f"{type(self)} must override process_custom_jvp_call "
               "to handle custom_jvp primitives")
        raise NotImplementedError(msg)

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
        msg = (f"{type(self)} must override process_custom_vjp_call "
               "to handle custom_vjp primitives")
        raise NotImplementedError(msg)

def escaped_tracer_error(tracer, detail=None):
    num_frames = FLAGS.jax_tracer_error_num_traceback_frames
    msg = ("Encountered an unexpected tracer. Perhaps this tracer escaped "
           "through global state from a previously traced function.\n"
           "The functions being transformed should not save traced values to "
           "global state.")
    if detail:
        msg += " Detail: {}.".format(detail)
    try:
        line_info = tracer._line_info
    except AttributeError:
        pass
    else:
        msg += ('\nThe tracer that caused this error was created on line '
                f'{source_info_util.summarize(line_info)}. The tracer has'
                f' shape {tracer.shape} and dtype {tracer.dtype}.\n')
        if num_frames > 0:
            msg += (f'When the tracer was created, the final {num_frames} stack '
                    'frames (most recent last) excluding JAX-internal frames were:\n'
                    f'{source_info_util.summarize(line_info, num_frames=num_frames)}')
    dbg = getattr(tracer._trace.main, 'debug_info', None)
    if dbg is not None:
        msg += ('\nThe function being traced when the tracer leaked was '
                f'{dbg.func_src_info} traced for {dbg.traced_for}.')
    msg += ('\nTo catch the leak earlier, try setting the environment variable '
            'JAX_CHECK_TRACER_LEAKS or using the `jax.checking_leaks` context '
            'manager.')
    return errors.LispUnexpectedTracerError(msg)




from dataclasses import dataclass


@dataclass
class _LispTracerBase:
    _trace: LispTrace
    __slots__ = ['_trace', '__weakref__', '_line_info']


class LispTracer(_LispTracerBase):
    __array_priority__ = 1000

    def __array__(self, *args, **kw):
        raise errors.LispTracerArrayConversionError(self)

    def __index__(self):
        raise errors.LispTracerIntegerConversionError(self)

    def __init__(self, trace: LispTrace):
        self._trace = trace

    def __iter__(self):
        return iter(self.aval._iter(self))

    def __len__(self):
        return self.aval._len(self)

    @property
    def aval(self):
        raise NotImplementedError("must override")

    def _assert_live(self) -> None:
        pass  # Override for liveness checking

    # Python looks up special methods only on classes, not instances. This means
    # these methods needs to be defined explicitly rather than relying on
    # __getattr__.
    def __neg__(self): return self.aval._neg(self)
    def __pos__(self): return self.aval._pos(self)
    def __eq__(self, other): return self.aval._eq(self, other)
    def __ne__(self, other): return self.aval._ne(self, other)
    def __lt__(self, other): return self.aval._lt(self, other)
    def __le__(self, other): return self.aval._le(self, other)
    def __gt__(self, other): return self.aval._gt(self, other)
    def __ge__(self, other): return self.aval._ge(self, other)
    def __abs__(self): return self.aval._abs(self)
    def __add__(self, other): return self.aval._add(self, other)
    def __radd__(self, other): return self.aval._radd(self, other)
    def __sub__(self, other): return self.aval._sub(self, other)
    def __rsub__(self, other): return self.aval._rsub(self, other)
    def __mul__(self, other): return self.aval._mul(self, other)
    def __rmul__(self, other): return self.aval._rmul(self, other)
    def __div__(self, other): return self.aval._div(self, other)
    def __rdiv__(self, other): return self.aval._rdiv(self, other)
    def __truediv__(self, other): return self.aval._truediv(self, other)
    def __rtruediv__(self, other): return self.aval._rtruediv(self, other)
    def __floordiv__(self, other): return self.aval._floordiv(self, other)
    def __rfloordiv__(self, other): return self.aval._rfloordiv(self, other)
    def __divmod__(self, other): return self.aval._divmod(self, other)
    def __rdivmod__(self, other): return self.aval._rdivmod(self, other)
    def __mod__(self, other): return self.aval._mod(self, other)
    def __rmod__(self, other): return self.aval._rmod(self, other)
    def __pow__(self, other): return self.aval._pow(self, other)
    def __rpow__(self, other): return self.aval._rpow(self, other)
    def __matmul__(self, other): return self.aval._matmul(self, other)
    def __rmatmul__(self, other): return self.aval._rmatmul(self, other)
    def __and__(self, other): return self.aval._and(self, other)
    def __rand__(self, other): return self.aval._rand(self, other)
    def __or__(self, other): return self.aval._or(self, other)
    def __ror__(self, other): return self.aval._ror(self, other)
    def __xor__(self, other): return self.aval._xor(self, other)
    def __rxor__(self, other): return self.aval._rxor(self, other)
    def __invert__(self): return self.aval._invert(self)
    def __lshift__(self, other): return self.aval._lshift(self, other)
    def __rlshift__(self, other): return self.aval._rlshift(self, other)
    def __rshift__(self, other): return self.aval._rshift(self, other)
    def __rrshift__(self, other): return self.aval._rrshift(self, other)
    def __getitem__(self, idx): return self.aval._getitem(self, idx)
    def __nonzero__(self): return self.aval._nonzero(self)
    def __bool__(self): return self.aval._bool(self)
    def __int__(self): return self.aval._int(self)
    def __long__(self): return self.aval._long(self)
    def __hex__(self): return self.aval._hex(self)
    def __oct__(self): return self.aval._oct(self)
    def __float__(self): return self.aval._float(self)
    def __complex__(self): return self.aval._complex(self)

    def __setitem__(self, idx, val):
        raise TypeError("JAX 'LispTracer' objects do not support item assignment")

    # NumPy also only looks up special methods on classes.
    def __array_module__(self, types): return self.aval._array_module(self, types)

    def __getattr__(self, name):
        # if the aval property raises an AttributeError, gets caught here
        assert not config.jax_enable_checks or name != "aval"

        try:
            attr = getattr(self.aval, name)
        except KeyError as err:
            raise AttributeError(
                "{} has no attribute {}".format(self.__class__.__name__, name)
            ) from err
        else:
            t = type(attr)
            if t is aval_property:
                return attr.fget(self)
            elif t is aval_method:
                return types.MethodType(attr.fun, self)
            else:
                return attr

    def __repr__(self):
        base = pp('Traced<{}>with<{}>'.format(self.aval, self._trace))
        contents = [(name, pp(repr(attr))) for name, attr in self._contents()]
        if contents:
            base += pp('  with ') >> vcat(pp('{} = '.format(name)) >> pp_payload
                                          for name, pp_payload in contents)
        return str(base)

    def _contents(self):
        try:
            return [(name, getattr(self, name)) for name in self.__slots__]
        except AttributeError:
            return ()

    def __copy__(self):
        return self

    def __deepcopy__(self, unused_memo):
        return self

    def _origin_msg(self) -> str:
        return ""


# -------------------- abstract values --------------------


class AbstractValue:
    __slots__: List[str] = []
    _num_buffers: int = 1  # number of buffers used to represent the value.

    def at_least_vspace(self):
        raise NotImplementedError("must override")

    def __repr__(self):
        try:
            kv_pairs = ('{}={}'.format(k, v) for k, v in self.__dict__.items())
            return '{}({})'.format(self.__class__.__name__, ','.join(kv_pairs))
        except AttributeError:
            return self.__class__.__name__

    def strip_weak_type(self) -> 'AbstractValue':
        return self

    def strip_named_shape(self) -> 'AbstractValue':
        return self

    def join(self, other):
        raise NotImplementedError("must override")

    def update(self, **kwargs):
        raise NotImplementedError("must override")

    def str_short(self):
        raise NotImplementedError("must override")


def concretization_function_error(fun, suggest_astype=False):
  fname = getattr(fun, "__name__", fun)
  fname_context = f"The problem arose with the `{fname}` function. "
  if suggest_astype:
    fname_context += ("If trying to convert the data type of a value, "
                      f"try using `x.astype({fun.__name__})` "
                      f"or `jnp.array(x, {fun.__name__})` instead.")
  def error(self, arg):
    raise errors.LispConcretizationTypeError(arg, fname_context)
  return error

def concrete_or_error(force: Any, val: Any, context=""):
  """Like force(val), but gives the context in the error message."""
  if force is None:
    force = lambda x: x
  if isinstance(val, LispTracer):
    if isinstance(val.aval, ConcreteArray):
      return force(val.aval.val)
    else:
      raise errors.LispConcretizationTypeError(val, context)
  else:
    return force(val)


class UnshapedArray(AbstractValue):
    __slots__ = ['dtype', 'weak_type']
    array_abstraction_level = 2

    def __init__(self, dtype, weak_type=False):
        self.dtype = np.dtype(dtypes.canonicalize_dtype(dtype))
        self.weak_type = weak_type

    def update(self, dtype=None, weak_type=None):
        if dtype is None:
            dtype = self.dtype
        if weak_type is None:
            weak_type = self.weak_type
        return UnshapedArray(dtype, weak_type)

    def __eq__(self, other):
        return (type(self) is type(other) and self.dtype == other.dtype and
                self.weak_type == other.weak_type)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
        # objects, e.g. `np.zeros(3).dtype is np.zeros(4).dtype`, or we can use
        # the unique character code via hash(self.dtype.char)
        return hash((self.dtype, self.weak_type))

    def __repr__(self):
        return '{}({}{})'.format(self.__class__.__name__, self.str_short(),
                                 ", weak_type=True" if self.weak_type else "")

    _bool = _nonzero = concretization_function_error(bool)
    _float   = concretization_function_error(float, True)
    _int     = concretization_function_error(int, True)
    _complex = concretization_function_error(complex, True)
    _hex     = concretization_function_error(hex)
    _oct     = concretization_function_error(oct)

    def at_least_vspace(self) -> AbstractValue:
        return UnshapedArray(primal_dtype_to_tangent_dtype(self.dtype),
                             self.weak_type)

    def join(self, other):
        if self.dtype == other.dtype:
            if self.weak_type == other.weak_type:
                return self
            else:
                return UnshapedArray(self.dtype, weak_type=False)
        else:
            raise TypeError(self, other)

    def str_short(self) -> str:
        return self.dtype.name

    def strip_weak_type(self):
        """Returns a copy of the aval with weak_type=False."""
        return self.update(weak_type=False)

    @property
    def shape(self):
        msg = ("UnshapedArray has no shape. Please open an issue at "
               "https://github.com/google/jax/issues because it's unexpected for "
               "UnshapedArray instances to ever be produced.")
        raise TypeError(msg)

class ShapedArray(UnshapedArray):
    __slots__ = ['shape', 'named_shape']
    array_abstraction_level = 1

    def __init__(self, shape, dtype, weak_type=False, named_shape={}):
        super(ShapedArray, self).__init__(dtype, weak_type=weak_type)
        self.shape = canonicalize_shape(shape)
        self.named_shape = dict(named_shape)

    def update(self, shape=None, dtype=None, weak_type=None, named_shape=None):
        if shape is None:
            shape = self.shape
        if dtype is None:
            dtype = self.dtype
        if weak_type is None:
            weak_type = self.weak_type
        if named_shape is None:
            named_shape = self.named_shape
        return ShapedArray(shape, dtype, weak_type, named_shape)

    ndim = property(lambda self: len(self.shape))
    size = property(lambda self: prod(self.shape))

    broadcast: ClassVar[Optional[aval_method]] = None
    transpose: ClassVar[Optional[aval_method]] = None
    reshape: ClassVar[Optional[aval_method]] = None
    _iter: ClassVar[Optional[staticmethod]] = None

    def __eq__(self, other):
        return (type(self) is type(other)
                and self.dtype == other.dtype and self.shape == other.shape
                and self.weak_type == other.weak_type
                and self.named_shape == other.named_shape)

    def __hash__(self):
        # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
        # objects, e.g. `np.zeros(3).dtype is np.zeros(4).dtype`, or we can use
        # the unique character code via hash(self.dtype.char)
        return hash((self.shape, self.dtype, self.weak_type,
                     tuple(self.named_shape.items())))

    def at_least_vspace(self):
        return ShapedArray(self.shape, primal_dtype_to_tangent_dtype(self.dtype),
                           self.weak_type, self.named_shape)

    def join(self, other):
        if symbolic_equal_shape(self.shape, other.shape) and self.dtype == other.dtype:
            weak_type = self.weak_type and other.weak_type
            named_shape = join_named_shapes(self.named_shape, other.named_shape)
            return self.update(weak_type=weak_type, named_shape=named_shape)
        elif self.dtype == other.dtype:
            return UnshapedArray(self.dtype)
        else:
            raise TypeError(self, other)

    def str_short(self):
        shapestr = ','.join(map(str, self.shape))
        if self.named_shape:
            named_shapestr = ','.join(f'{k}:{v}' for k, v in self.named_shape.items())
            return f'{self.dtype.name}[{shapestr};{named_shapestr}]'
        else:
            return f'{self.dtype.name}[{shapestr}]'

    def strip_named_shape(self):
        return self.update(named_shape={})

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError as err:
            raise TypeError("len() of unsized object") from err  # same as numpy error

    def _len(self, ignored_tracer):
        return len(self)


def _forward_to_value(self, fun, ignored_tracer, *args):
    return fun(self.val, *args)

class ConcreteArray(ShapedArray):
    __slots__ = ['val']
    array_abstraction_level = 0

    def __init__(self, val, weak_type=False):
        super(ConcreteArray, self).__init__(np.shape(val), np.result_type(val),
                                            weak_type=weak_type)
        # Note: canonicalized self.dtype doesn't necessarily match self.val
        self.val = val
        assert self.dtype != np.dtype('O'), val

    def update(self, val=None, weak_type=None):
        if val is None:
            val = self.val
        if weak_type is None:
            weak_type = self.weak_type
        return ConcreteArray(val, weak_type)

    def __eq__(self, other):
        if (type(self) is type(other) and self.dtype == other.dtype
                and self.shape == other.shape and self.weak_type == other.weak_type):
            with eval_context():  # in case self.val is a DeviceArray
                return (self.val == other.val).all()
        else:
            return False

    def __hash__(self):
        return id(self.val)

    def join(self, other) -> AbstractValue:
        if self == other:
            return self
        elif self.shape == other.shape and self.dtype == other.dtype:
            weak_type = self.weak_type and other.weak_type
            named_shape = join_named_shapes(self.named_shape, other.named_shape)
            return ShapedArray(
                self.shape, self.dtype, weak_type=weak_type, named_shape=named_shape)
        elif self.dtype == other.dtype:
            return UnshapedArray(self.dtype,
                                 weak_type=self.weak_type and other.weak_type)
        else:
            raise TypeError(self, other)

    def str_short(self) -> str:
        return str(self.val)

    _bool = _nonzero = partialmethod(_forward_to_value, bool)
    _int             = partialmethod(_forward_to_value, int)
    _hex             = partialmethod(_forward_to_value, hex)
    _oct             = partialmethod(_forward_to_value, oct)

    _float           = concretization_function_error(float, True)
    _complex         = concretization_function_error(complex, True)
