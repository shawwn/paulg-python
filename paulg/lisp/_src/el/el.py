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

# Pytype is too slow to check this file.
# pytype: skip-file

import builtins
from enum import IntEnum
import functools
import itertools
import operator
from typing import (Any, Callable, List, NamedTuple, Optional, Sequence,\
                    Union, Tuple)
import warnings

import numpy as np

from paulg import lisp
from paulg.lisp import core
# from paulg.lisp._src import ad_util
# from paulg.lisp._src import api
# from paulg.lisp import api_util
# from paulg.lisp import linear_util as lu
# from paulg.lisp._src import dtypes
# from paulg.lisp import tree_util
# from paulg.lisp._src.config import config
from paulg.lisp.core import LispPrimitive
# from paulg.lisp.core import (Primitive, _canonicalize_dimension, UnshapedArray,
#                       ShapedArray, ConcreteArray, raise_to_shaped,
#                       abstract_token, canonicalize_shape)
# from paulg.lisp._src.abstract_arrays import array_types
# from paulg.lisp.interpreters import partial_eval as pe
# from paulg.lisp.interpreters import xla
# from paulg.lisp.interpreters import pxla
# from paulg.lisp.interpreters import ad
# from paulg.lisp.interpreters import invertible_ad as iad
# from paulg.lisp.interpreters import batching
# from paulg.lisp.interpreters import masking
from paulg.lisp._src.util import (cache, safe_zip, partial, prod, safe_map,
                           canonicalize_axis, split_list)
# from paulg.lisp.tree_util import tree_map
# from paulg.lisp.lib import pytree
# from paulg.lisp.lib import xla_bridge
# from paulg.lisp.lib import xla_client

# xb = xla_bridge
# xc = xla_client
# xops = xla_client.ops

_max = builtins.max
_min = builtins.min
_reduce = functools.reduce

Array = Any
DType = Any
Shape = core.Shape

def _try_broadcast_shapes(shapes):
  assert shapes
  if len(shapes) == 1: return shapes[0]
  rank, *others = {len(shape) for shape in shapes}
  if others: return None  # must have consistent rank
  if not rank: return ()  # scalar case
  result_shape = [None] * rank
  for i, sizes in enumerate(zip(*shapes)):
    non_1s = set([d for d in sizes if not core.symbolic_equal_dim(d, 1)])
    if len(non_1s) > 1:
      return None  # must have equal sizes other than 1-sized axes
    result_shape[i] = next(iter(non_1s), 1)

  return tuple(result_shape)

@cache()
def broadcast_shapes(*shapes):
  """Returns the shape that results from NumPy broadcasting of `shapes`."""
  if len(shapes) == 1:
    return shapes[0]
  ndim = _max(len(shape) for shape in shapes)
  shapes = [(1,) * (ndim - len(shape)) + shape for shape in shapes]
  result_shape = _try_broadcast_shapes(shapes)
  if result_shape is None:
    raise ValueError("Incompatible shapes for broadcasting: {}"
                     .format(tuple(map(tuple, shapes))))
  return result_shape

def _identity(x): return x


def standard_primitive(shape_rule, dtype_rule, name, translation_rule=None,
                       weak_type_rule=None, named_shape_rule=None):
  weak_type_rule = weak_type_rule or _standard_weak_type_rule
  named_shape_rule = named_shape_rule or standard_named_shape_rule
  prim = LispPrimitive(name)
  # prim.def_impl(partial(xla.apply_primitive, prim))
  prim.def_abstract_eval(
      partial(standard_abstract_eval, prim, shape_rule, dtype_rule,
              weak_type_rule, named_shape_rule))
  xla.translations[prim] = translation_rule or partial(standard_translate, name)
  return prim

def standard_abstract_eval(prim, shape_rule, dtype_rule, weak_type_rule,
                           named_shape_rule, *avals, **kwargs):
  assert all(isinstance(aval, UnshapedArray) for aval in avals), avals
  assert not prim.multiple_results
  weak_type = weak_type_rule(*avals, **kwargs)
  least_specialized = _max(map(type, avals),
                           key=operator.attrgetter('array_abstraction_level'))
  if least_specialized is ConcreteArray:
    return ConcreteArray(prim.impl(*[x.val for x in avals], **kwargs),
                         weak_type=weak_type)
  elif least_specialized is ShapedArray:
    return ShapedArray(shape_rule(*avals, **kwargs), dtype_rule(*avals, **kwargs),
                       weak_type=weak_type,
                       named_shape=named_shape_rule(*avals, **kwargs))
  elif least_specialized is UnshapedArray:
    return UnshapedArray(dtype_rule(*avals, **kwargs), weak_type=weak_type)
  else:
    raise TypeError(avals, least_specialized)

def standard_translate(name, c, *args, **kwargs):
  xla_opname = ''.join(term.capitalize() for term in name.split('_'))
  return getattr(xops, xla_opname)(*args, **kwargs)

def standard_named_shape_rule(*avals, **kwargs):
  return core.join_named_shapes(*(a.named_shape for a in avals))

def _standard_weak_type_rule(*avals, **kwargs):
  return all(aval.weak_type for aval in avals)


### traceables

def neg(x: Array) -> Array:
  r"""Elementwise negation: :math:`-x`."""
  return neg_p.bind(x)



_float = {np.floating}
_complex = {np.complexfloating}
_complex_elem_types = {np.float32, np.float64}
_int = {np.integer}
_bool = {np.bool_}

_num = _int | _float | _complex
_any = _int | _float | _complex | _bool
_bool_or_int = _int | _bool
