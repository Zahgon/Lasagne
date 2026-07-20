import numbers

import numpy as np
import theano
import theano.tensor as T


int_types = (numbers.Integral, np.integer)


def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.

    Parameters
    ----------
    arr : array_like
        The data to be converted.

    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return np.asarray(arr, dtype=theano.config.floatX)


def shared_empty(dim=2, dtype=None):
    pass


def as_theano_expression(input):
    """Wrap as Theano expression.

    Wraps the given input as a Theano constant if it is not
    a valid Theano expression already. Useful to transparently
    handle numpy arrays and Python scalars, for example.

    Parameters
    ----------
    input : number, numpy array or Theano expression
        Expression to be converted to a Theano constant.

    Returns
    -------
    Theano symbolic constant
        Theano constant version of `input`.
    """
    if isinstance(input, theano.gof.Variable):
        return input
    else:
        try:
            return theano.tensor.constant(input)
        except Exception as e:
            raise TypeError("Input of type %s is not a Theano expression and "
                            "cannot be wrapped as a Theano constant (original "
                            "exception: %s)" % (type(input), e))


def collect_shared_vars(expressions):
    """Returns all shared variables the given expression(s) depend on.

    Parameters
    ----------
    expressions : Theano expression or iterable of Theano expressions
        The expressions to collect shared variables from.

    Returns
    -------
    list of Theano shared variables
        All shared variables the given expression(s) depend on, in fixed order
        (as found by a left-recursive depth-first search). If some expressions
        are shared variables themselves, they are included in the result.
    """
    if isinstance(expressions, theano.Variable):
        expressions = [expressions]
    return [v for v in theano.gof.graph.inputs(reversed(expressions))
            if isinstance(v, theano.compile.SharedVariable)]


def one_hot(x, m=None):
    pass


def unique(l):
    """Filters duplicates of iterable.

    Create a new list from l with duplicate entries removed,
    while preserving the original order.

    Parameters
    ----------
    l : iterable
        Input iterable to filter of duplicates.

    Returns
    -------
    list
        A list of elements of `l` without duplicates and in the same order.
    """
    new_list = []
    seen = set()
    for el in l:
        if el not in seen:
            new_list.append(el)
            seen.add(el)

    return new_list


def as_tuple(x, N, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).

    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type or tuple of type, optional
        required type or types for all elements

    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.

    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it
    ValueError
        if `x` is iterable, but does not have exactly `N` elements
    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        if t == int_types:
            expected_type = "int"  # easier to understand
        elif isinstance(t, tuple):
            expected_type = " or ".join(tt.__name__ for tt in t)
        else:
            expected_type = t.__name__
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(expected_type, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X


def inspect_kwargs(func):
    """
    Inspects a callable and returns a list of all optional keyword arguments.

    Parameters
    ----------
    func : callable
        The callable to inspect

    Returns
    -------
    kwargs : list of str
        Names of all arguments of `func` that have a default value, in order
    """
    try:
        from inspect import signature
    except ImportError:  # pragma: no cover
        from inspect import getargspec
        spec = getargspec(func)
        return spec.args[-len(spec.defaults):] if spec.defaults else []
    else:  # pragma: no cover
        params = signature(func).parameters
        return [p.name for p in params.values() if p.default is not p.empty]


def compute_norms(array, norm_axes=None):
    pass


def create_param(spec, shape, name=None):
    pass


def unroll_scan(fn, sequences, outputs_info, non_sequences, n_steps,
                go_backwards=False):
        """
        Helper function to unroll for loops. Can be used to unroll theano.scan.
        The parameter names are identical to theano.scan, please refer to here
        for more information.

        Note that this function does not support the truncate_gradient
        setting from theano.scan.

        Parameters
        ----------

        fn : function
            Function that defines calculations at each step.

        sequences : TensorVariable or list of TensorVariables
            List of TensorVariable with sequence data. The function iterates
            over the first dimension of each TensorVariable.

        outputs_info : list of TensorVariables
            List of tensors specifying the initial values for each recurrent
            value.

        non_sequences: list of TensorVariables
            List of theano.shared variables that are used in the step function.

        n_steps: int
            Number of steps to unroll.

        go_backwards: bool
            If true the recursion starts at sequences[-1] and iterates
            backwards.

        Returns
        -------
        List of TensorVariables. Each element in the list gives the recurrent
        values at each time step.

        """
        if not isinstance(sequences, (list, tuple)):
            sequences = [sequences]

        counter = range(n_steps)
        if go_backwards:
            counter = counter[::-1]

        output = []
        prev_vals = outputs_info
        for i in counter:
            step_input = [s[i] for s in sequences] + prev_vals + non_sequences
            out_ = fn(*step_input)
            if isinstance(out_, T.TensorVariable):
                out_ = [out_]
            if isinstance(out_, tuple):
                out_ = list(out_)
            output.append(out_)

            prev_vals = output[-1]

        output_scan = []
        for i in range(len(output[0])):
            l = map(lambda x: x[i], output)
            output_scan.append(T.stack(*l))

        return output_scan
