
class _LispErrorMixin:
    """Mixin for Lisp-specific errors"""
    _error_page = 'https://paulg.readthedocs.io/en/latest/errors.html'
    _module_name = "paulg.lisp.errors"

    def __init__(self, message: str):
        error_page = self._error_page
        module_name = self._module_name
        class_name = self.__class__.__name__
        error_msg = f'{message}\nSee {error_page}#{module_name}.{class_name}'
        # https://github.com/python/mypy/issues/5887
        super().__init__(error_msg)  # type: ignore


class LispTypeError(_LispErrorMixin, TypeError):
    pass


class LispIndexError(_LispErrorMixin, IndexError):
    pass


class LispConcretizationTypeError(LispTypeError):
  """
  This error occurs when a Lisp Tracer object is used in a context where a
  concrete value is required. In some situations, it can be easily fixed by
  marking problematic values as static; in others, it may indicate that your
  program is doing operations that are not directly supported by Lisp's JIT
  compilation model.

  Traced value where static value is expected
    One common cause of this error is using a traced value where a static value
    is required. For example:

      >>> from jax import jit, partial
      >>> import jax.numpy as jnp
      >>> @jit
      ... def func(x, axis):
      ...   return x.min(axis)

      >>> func(jnp.arange(4), 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      ConcretizationTypeError: Abstract tracer value encountered where concrete
      value is expected: axis argument to jnp.min().

    This can often be fixed by marking the problematic argument as static::

        >>> @partial(jit, static_argnums=1)
        ... def func(x, axis):
        ...   return x.min(axis)

        >>> func(jnp.arange(4), 0)
        DeviceArray(0, dtype=int32)

  Traced value used in control flow
    Another case where this often arises is when a traced value is used in
    Python control flow. For example::

      >>> @jit
      ... def func(x, y):
      ...   return x if x.sum() < y.sum() else y

      >>> func(jnp.ones(4), jnp.zeros(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      ConcretizationTypeError: Abstract tracer value encountered where concrete
      value is expected: [...]

    We could mark both inputs ``x`` and ``y`` as static, but that would defeat
    the purpose of using :func:`jax.jit` here. Another option is to re-express
    the if statement in terms of :func:`jax.numpy.where`::

      >>> @jit
      ... def func(x, y):
      ...   return jnp.where(x.sum() < y.sum(), x, y)

      >>> func(jnp.ones(4), jnp.zeros(4))
      DeviceArray([0., 0., 0., 0.], dtype=float32)

    For more complicated control flow including loops, see
    :ref:`lax-control-flow`.

  Shape depends on Traced Value
    Such an error may also arise when a shape in your JIT-compiled computation
    depends on the values within a traced quantity. For example::

      >>> @jit
      ... def func(x):
      ...     return jnp.where(x < 0)

      >>> func(jnp.arange(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected:
      The error arose in jnp.nonzero.

    This is an example of an operation that is incompatible with Lisp's JIT
    compilation model, which requires array sizes to be known at compile-time.
    Here the size of the returned array depends on the contents of `x`, and such
    code cannot be JIT compiled.

    In many cases it is possible to work around this by modifying the logic used
    in the function; for example here is code with a similar issue::

      >>> @jit
      ... def func(x):
      ...     indices = jnp.where(x > 1)
      ...     return x[indices].sum()

      >>> func(jnp.arange(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      ConcretizationTypeError: Abstract tracer value encountered where concrete
      value is expected: The error arose in jnp.nonzero.

    And here is how you might express the same operation in a way that avoids
    creation of a dynamically-sized index array::

      >>> @jit
      ... def func(x):
      ...   return jnp.where(x > 1, x, 0).sum()

      >>> func(jnp.arange(4))
      DeviceArray(5, dtype=int32)

  To understand more subtleties having to do with tracers vs. regular values,
  and concrete vs. abstract values, you may want to read
  :ref:`faq-different-kinds-of-jax-values`.
  """
  def __init__(self, tracer: "core.Tracer", context: str = ""):
    super().__init__(
        "Abstract tracer value encountered where concrete value is expected: "
        f"{tracer}\n{context}{tracer._origin_msg()}\n")


class LispNonConcreteBooleanIndexError(LispIndexError):
  """
  This error occurs when a program attempts to use non-concrete boolean indices
  in a traced indexing operation. Under JIT compilation, Lisp arrays must have
  static shapes (i.e. shapes that are known at compile-time) and so boolean
  masks must be used carefully. Some logic implemented via boolean masking is
  simply not possible in a :func:`jax.jit` function; in other cases, the logic
  can be re-expressed in a JIT-compatible way, often using the three-argument
  version of :func:`~jax.numpy.where`.

  Following are a few examples of when this error might arise.

  Constructing arrays via boolean masking
    This most commonly arises when attempting to create an array via a boolean
    mask within a JIT context. For example::

      >>> import jax
      >>> import jax.numpy as jnp

      >>> @jax.jit
      ... def positive_values(x):
      ...   return x[x > 0]

      >>> positive_values(jnp.arange(-5, 5))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      NonConcreteBooleanIndexError: Array boolean indices must be concrete: ShapedArray(bool[10])

    This function is attempting to return only the positive values in the input
    array; the size of this returned array cannot be determined at compile-time
    unless `x` is marked as static, and so operations like this cannot be
    performed under JIT compilation.

  Reexpressible Boolean Logic
    Although creating dynamically sized arrays is not supported directly, in
    many cases it is possible to re-express the logic of the computation in
    terms of a JIT-compatible operation. For example, here is another function
    that fails under JIT for the same reason::

      >>> @jax.jit
      ... def sum_of_positive(x):
      ...   return x[x > 0].sum()

      >>> sum_of_positive(jnp.arange(-5, 5))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      NonConcreteBooleanIndexError: Array boolean indices must be concrete: ShapedArray(bool[10])

    In this case, however, the problematic array is only an intermediate value,
    and we can instead express the same logic in terms of the JIT-compatible
    three-argument version of :func:`jax.numpy.where`::

      >>> @jax.jit
      ... def sum_of_positive(x):
      ...   return jnp.where(x > 0, x, 0).sum()

      >>> sum_of_positive(jnp.arange(-5, 5))
      DeviceArray(10, dtype=int32)

    This pattern of replacing boolean masking with three-argument
    :func:`~jax.numpy.where` is a common solution to this sort of problem.

  Boolean indices in :mod:`jax.ops`
    The other situation where this error often arises is when using boolean
    indices within functions in :mod:`jax.ops`, such as
    :func:`jax.ops.index_update`. Here is a simple example::

      >>> @jax.jit
      ... def manual_clip(x):
      ...   return jax.ops.index_update(x, x < 0, 0)

      >>> manual_clip(jnp.arange(-2, 2))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      NonConcreteBooleanIndexError: Array boolean indices must be concrete: ShapedArray(bool[4])

    This function is attempting to set values smaller than zero to a scalar fill
    value. As above, this can be addressed by re-expressing the logic in terms
    of :func:`~jax.numpy.where`::

      >>> @jax.jit
      ... def manual_clip(x):
      ...   return jnp.where(x < 0, 0, x)

      >>> manual_clip(jnp.arange(-2, 2))
      DeviceArray([0, 0, 0, 1], dtype=int32)

    These operations also commonly are written in terms of the
    :ref:`syntactic-sugar-for-ops`; for example, this is syntactic sugar for
    :func:`~jax.ops.index_mul`, and fails under JIT::

      >>> @jax.jit
      ... def manual_abs(x):
      ...   return x.at[x < 0].mul(-1)

      >>> manual_abs(jnp.arange(-2, 2))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      NonConcreteBooleanIndexError: Array boolean indices must be concrete: ShapedArray(bool[4])

    As above, the solution is to re-express this in terms of
    :func:`~jax.numpy.where`::

      >>> @jax.jit
      ... def manual_abs(x):
      ...   return jnp.where(x < 0, x * -1, x)

      >>> manual_abs(jnp.arange(-2, 2))
      DeviceArray([2, 1, 0, 1], dtype=int32)
  """
  def __init__(self, tracer: "core.Tracer"):
    super().__init__(
        f"Array boolean indices must be concrete; got {tracer}\n")


class LispTracerArrayConversionError(LispTypeError):
  """
  This error occurs when a program attempts to convert a Lisp Tracer object into
  a standard NumPy array. It typically occurs in one of a few situations.

  Using `numpy` rather than `jax.numpy` functions
    This error can occur when a Lisp Tracer object is passed to a raw numpy
    function, or a method on a numpy.ndarray object. For example::

      >>> from jax import jit, partial
      >>> import numpy as np
      >>> import jax.numpy as jnp

      >>> @jit
      ... def func(x):
      ...   return np.sin(x)

      >>> func(jnp.arange(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerArrayConversionError: The numpy.ndarray conversion method
      __array__() was called on the Lisp Tracer object

    In this case, check that you are using `jax.numpy` methods rather than
    `numpy` methods::

      >>> @jit
      ... def func(x):
      ...   return jnp.sin(x)

      >>> func(jnp.arange(4))
      DeviceArray([0.        , 0.84147096, 0.9092974 , 0.14112   ], dtype=float32)

  Indexing a numpy array with a tracer
    If this error arises on a line that involves array indexing, it may be that
    the array being indexed `x` is a raw numpy.ndarray while the indices `idx`
    are traced. For example::

      >>> x = np.arange(10)

      >>> @jit
      ... def func(i):
      ...   return x[i]

      >>> func(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerArrayConversionError: The numpy.ndarray conversion method
      __array__() was called on the Lisp Tracer object

    Depending on the context, you may fix this by converting the numpy array
    into a Lisp array::

      >>> @jit
      ... def func(i):
      ...   return jnp.asarray(x)[i]

      >>> func(0)
      DeviceArray(0, dtype=int32)

    or by declaring the index as a static argument::

      >>> @partial(jit, static_argnums=(0,))
      ... def func(i):
      ...   return x[i]

      >>> func(0)
      DeviceArray(0, dtype=int32)

  To understand more subtleties having to do with tracers vs. regular values,
  and concrete vs. abstract values, you may want to read
  :ref:`faq-different-kinds-of-jax-values`.
  """
  def __init__(self, tracer: "core.Tracer"):
    super().__init__(
        "The numpy.ndarray conversion method __array__() was called on "
        f"the Lisp Tracer object {tracer}{tracer._origin_msg()}")


class LispTracerIntegerConversionError(LispTypeError):
  """
  This error can occur when a Lisp Tracer object is used in a context where a
  Python integer is expected. It typically occurs in a few situations.

  Passing a tracer in place of an integer
    This error can occur if you attempt to pass a tracer to a function that
    requires an integer argument; for example::

      >>> from jax import jit, partial
      >>> import numpy as np

      >>> @jit
      ... def func(x, axis):
      ...   return np.split(x, 2, axis)

      >>> func(np.arange(4), 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerIntegerConversionError: The __index__() method was called on the Lisp
      Tracer object

    When this happens, the solution is often to mark the problematic argument as
    static::

      >>> @partial(jit, static_argnums=1)
      ... def func(x, axis):
      ...   return np.split(x, 2, axis)

      >>> func(np.arange(10), 0)
      [DeviceArray([0, 1, 2, 3, 4], dtype=int32),
       DeviceArray([5, 6, 7, 8, 9], dtype=int32)]

    An alternative is to apply the transformation to a closure that encapsulates
    the arguments to be protected, either manually as below or by using
    :func:`functools.partial`::

      >>> jit(lambda arr: np.split(arr, 2, 0))(np.arange(4))
      [DeviceArray([0, 1], dtype=int32), DeviceArray([2, 3], dtype=int32)]

    **Note a new closure is created at every invocation, which defeats the
    compilation caching mechanism, which is why static_argnums is preferred.**

  Indexing a list with a Tracer
    This error can occur if you attempt to index a Python list with a traced
    quantity.
    For example::

      >>> import jax.numpy as jnp
      >>> from jax import jit, partial

      >>> L = [1, 2, 3]

      >>> @jit
      ... def func(i):
      ...   return L[i]

      >>> func(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerIntegerConversionError: The __index__() method was called on the Lisp Tracer object

    Depending on the context, you can generally fix this either by converting
    the list to a Lisp array::

      >>> @jit
      ... def func(i):
      ...   return jnp.array(L)[i]

      >>> func(0)
      DeviceArray(1, dtype=int32)

    or by declaring the index as a static argument::

      >>> @partial(jit, static_argnums=0)
      ... def func(i):
      ...   return L[i]

      >>> func(0)
      DeviceArray(1, dtype=int32)

  To understand more subtleties having to do with tracers vs. regular values,
  and concrete vs. abstract values, you may want to read
  :ref:`faq-different-kinds-of-jax-values`.
  """
  def __init__(self, tracer: "core.Tracer"):
    super().__init__(
        f"The __index__() method was called on the Lisp Tracer object {tracer}")


class LispUnexpectedTracerError(LispTypeError):
  """
  This error occurs when you use a Lisp value that has leaked out of a function.
  What does it mean to leak a value? If you use a Lisp transformation on a
  function ``f`` that stores, in some scope outside of ``f``, a reference to
  an intermediate value, that value is considered to have been leaked.
  Leaking values is a side effect. (Read more about avoiding side effects in
  `Pure Functions <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions>`_)

  Lisp detects leaks when you then use the leaked value in another
  operation later on, at which point it raises an ``UnexpectedTracerError``.
  To fix this, avoid side effects: if a function computes a value needed
  in an outer scope, return that value from the transformed function explictly.

  Specifically, a ``Tracer`` is Lisp's internal representation of a function's
  intermediate values during transformations, e.g. within ``jit``, ``pmap``,
  ``vmap``, etc. Encountering a ``Tracer`` outside of a transformation implies a
  leak.

  Life-cycle of a leaked value
    Consider the following example of a transformed function which leaks a value
    to an outer scope::

      >>> from jax import jit
      >>> import jax.numpy as jnp

      >>> outs = []
      >>> @jit                   # 1
      ... def side_effecting(x):
      ...   y = x+1              # 3
      ...   outs.append(y)       # 4

      >>> x = 1
      >>> side_effecting(x)      # 2
      >>> outs[0]+1              # 5  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      UnexpectedTracerError: Encountered an unexpected tracer.

    In this example we leak a Traced value from an inner transformed scope to an
    outer scope. We get an ``UnexpectedTracerError`` when the leaked value is
    used, not when the value is leaked.

    This example also demonstrates the life-cycle of a leaked value:

      1. A function is transformed (in this case, by ``jit``)
      2. The transformed function is called (initiating an abstract trace of the
         function and turning ``x`` into a ``Tracer``)
      3. The intermediate value ``y``, which will later be leaked, is created
         (an intermediate value of a traced function is also a ``Tracer``)
      4. The value is leaked (appended to a list in an outer scope, escaping
         the function through a side-channel)
      5. The leaked value is used, and an UnexpectedTracerError is raised.

    The UnexpectedTracerError message tries to point to these locations in your
    code by including information about each stage. Respectively:

      1. The name of the transformed function (``side_effecting``) and which
         transform kicked of the trace (``jit``).
      2. A reconstructed stack trace of where the leaked Tracer was created,
         which includes where the transformed function was called.
         (``When the Tracer was created, the final 5 stack frames were...``).
      3. From the reconstructed stack trace, the line of code that created
         the leaked Tracer.
      4. The leak location is not included in the error message because it is
         difficult to pin down! Lisp can only tell you what the leaked value
         looks like (what shape is has and where it was created) and what
         boundary it was leaked over (the name of the transformation and the
         name of the transformed function).
      5. The current error's stack trace points to where the value is used.

    The error can be fixed by the returning the value out of the
    transformed function::

      >>> from jax import jit
      >>> import jax.numpy as jnp

      >>> outs = []
      >>> @jit
      ... def not_side_effecting(x):
      ...   y = x+1
      ...   return y

      >>> x = 1
      >>> y = not_side_effecting(x)
      >>> outs.append(y)
      >>> outs[0]+1  # all good! no longer a leaked value.
      DeviceArray(3, dtype=int32)

  Leak checker
    As discussed in point 2 and 3 above, Lisp shows a reconstructed stack trace
    which points to where the leaked value was created.  This is because
    Lisp only raises an error when the leaked value is used, not when the
    value is leaked. This is not the most useful place to raise this error,
    because you need to know the location where the Tracer was leaked to fix the
    error.

    To make this location easier to track down, you can use the leak checker.
    When the leak checker is enabled, an error is raised as soon as a ``Tracer``
    is leaked. (To be more exact, it will raise an error when the transformed
    function from which the ``Tracer`` is leaked returns)

    To enable the leak checker you can use the ``LISP_CHECK_TRACER_LEAKS``
    environment variable or the ``with jax.checking_leaks()`` context manager.

    .. note::
      Note that this tool is experimental and may report false positives. It
      works by disabling some Lisp caches, so it will have a negative effect on
      performance and should only be used when debugging.

    Example usage::

      >>> from jax import jit
      >>> import jax.numpy as jnp

      >>> outs = []
      >>> @jit
      ... def side_effecting(x):
      ...   y = x+1
      ...   outs.append(y)

      >>> x = 1
      >>> with jax.checking_leaks():
      ...   y = side_effecting(x)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      Exception: Leaked Trace

  """

  def __init__(self, msg: str):
    super().__init__(msg)
