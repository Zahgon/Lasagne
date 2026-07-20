from collections import deque
from difflib import get_close_matches
from itertools import chain
from warnings import warn

import theano
import numpy as np

from .. import utils


__all__ = [
    "get_all_layers",
    "get_output",
    "get_output_shape",
    "get_all_params",
    "count_params",
    "get_all_param_values",
    "set_all_param_values",
]


def get_all_layers(layer, treat_as_input=None):
    """
    This function gathers all layers below one or more given :class:`Layer`
    instances, including the given layer(s). Its main use is to collect all
    layers of a network just given the output layer(s). The layers are
    guaranteed to be returned in a topological order: a layer in the result
    list is always preceded by all layers its input depends on.

    Parameters
    ----------
    layer : Layer or list
        the :class:`Layer` instance for which to gather all layers feeding
        into it, or a list of :class:`Layer` instances.

    treat_as_input : None or iterable
        an iterable of :class:`Layer` instances to treat as input layers
        with no layers feeding into them. They will show up in the result
        list, but their incoming layers will not be collected (unless they
        are required for other layers as well).

    Returns
    -------
    list
        a list of :class:`Layer` instances feeding into the given
        instance(s) either directly or indirectly, and the given
        instance(s) themselves, in topological order.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)
    >>> get_all_layers(l1) == [l_in, l1]
    True
    >>> l2 = DenseLayer(l_in, num_units=10)
    >>> get_all_layers([l2, l1]) == [l_in, l2, l1]
    True
    >>> get_all_layers([l1, l2]) == [l_in, l1, l2]
    True
    >>> l3 = DenseLayer(l2, num_units=20)
    >>> get_all_layers(l3) == [l_in, l2, l3]
    True
    >>> get_all_layers(l3, treat_as_input=[l2]) == [l2, l3]
    True
    """
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()
    result = []

    if treat_as_input is not None:
        seen.update(treat_as_input)

    while queue:
        layer = queue[0]
        if layer is None:
            queue.popleft()
        elif layer not in seen:
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            queue.popleft()
            if layer not in done:
                result.append(layer)
                done.add(layer)

    return result


def get_output(layer_or_layers, inputs=None, **kwargs):
    """
    Computes the output of the network at one or more given layers.
    Optionally, you can define the input(s) to propagate through the network
    instead of using the input variable(s) associated with the network's
    input layer(s).

    Parameters
    ----------
    layer_or_layers : Layer or list
        the :class:`Layer` instance for which to compute the output
        expressions, or a list of :class:`Layer` instances.

    inputs : None, Theano expression, numpy array, or dict
        If None, uses the input variables associated with the
        :class:`InputLayer` instances.
        If a Theano expression, this defines the input for a single
        :class:`InputLayer` instance. Will throw a ValueError if there
        are multiple :class:`InputLayer` instances.
        If a numpy array, this will be wrapped as a Theano constant
        and used just like a Theano expression.
        If a dictionary, any :class:`Layer` instance (including the
        input layers) can be mapped to a Theano expression or numpy
        array to use instead of its regular output.

    Returns
    -------
    output : Theano expression or list
        the output of the given layer(s) for the given network input

    Notes
    -----
    Depending on your network architecture, `get_output([l1, l2])` may
    be crucially different from `[get_output(l1), get_output(l2)]`. Only
    the former ensures that the output expressions depend on the same
    intermediate expressions. For example, when `l1` and `l2` depend on
    a common dropout layer, the former will use the same dropout mask for
    both, while the latter will use two different dropout masks.
    """
    from .input import InputLayer
    from .base import MergeLayer, Layer
    if isinstance(inputs, dict):
        for input_key in inputs.keys():
            if (input_key is not None) and (not isinstance(input_key, Layer)):
                raise TypeError("The inputs dictionary keys must be"
                                " lasagne layers not %s." %
                                type(input_key))
    accepted_kwargs = {'deterministic'}
    treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    if isinstance(inputs, dict):
        all_outputs.update((layer, utils.as_theano_expression(expr))
                           for layer, expr in inputs.items())
    elif inputs is not None:
        if len(all_outputs) > 1:
            raise ValueError("get_output() was called with a single input "
                             "expression on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input expressions instead.")
        for input_layer in all_outputs:
            all_outputs[input_layer] = utils.as_theano_expression(inputs)
    for layer in all_layers:
        if layer not in all_outputs:
            try:
                if isinstance(layer, MergeLayer):
                    layer_inputs = [all_outputs[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    layer_inputs = all_outputs[layer.input_layer]
            except KeyError:
                raise ValueError("get_output() was called without giving an "
                                 "input expression for the free-floating "
                                 "layer %r. Please call it with a dictionary "
                                 "mapping this layer to an input expression."
                                 % layer)
            all_outputs[layer] = layer.get_output_for(layer_inputs, **kwargs)
            try:
                accepted_kwargs |= set(utils.inspect_kwargs(
                        layer.get_output_for))
            except TypeError:
                pass
            accepted_kwargs |= set(layer.get_output_kwargs)
    unused_kwargs = set(kwargs.keys()) - accepted_kwargs
    if unused_kwargs:
        suggestions = []
        for kwarg in unused_kwargs:
            suggestion = get_close_matches(kwarg, accepted_kwargs)
            if suggestion:
                suggestions.append('%s (perhaps you meant %s)'
                                   % (kwarg, suggestion[0]))
            else:
                suggestions.append(kwarg)
        warn("get_output() was called with unused kwargs:\n\t%s"
             % "\n\t".join(suggestions))
    try:
        return [all_outputs[layer] for layer in layer_or_layers]
    except TypeError:
        return all_outputs[layer_or_layers]


def get_output_shape(layer_or_layers, input_shapes=None):
    pass


def get_all_params(layer, unwrap_shared=True, **tags):
    """
    Returns a list of Theano shared variables or expressions that
    parameterize the layer.

    This function gathers all parameters of all layers below one or more given
    :class:`Layer` instances, including the layer(s) itself. Its main use is to
    collect all parameters of a network just given the output layer(s).

    By default, all shared variables that participate in the forward pass will
    be returned. The list can optionally be filtered by specifying tags as
    keyword arguments. For example, ``trainable=True`` will only return
    trainable parameters, and ``regularizable=True`` will only return
    parameters that can be regularized (e.g., by L2 decay).

    Parameters
    ----------
    layer : Layer or list
        The :class:`Layer` instance for which to gather all parameters, or a
        list of :class:`Layer` instances.

    unwrap_shared : bool (default: True)
        Affects only parameters that were set to a Theano expression. If
        ``True`` the function returns the shared variables contained in
        the expression, otherwise the Theano expression itself.

    **tags (optional)
        tags can be specified to filter the list. Specifying ``tag1=True``
        will limit the list to parameters that are tagged with ``tag1``.
        Specifying ``tag1=False`` will limit the list to parameters that
        are not tagged with ``tag1``. Commonly used tags are
        ``regularizable`` and ``trainable``.

    Returns
    -------
    params : list
        A list of Theano shared variables or expressions representing
        the parameters.

    Notes
    -----
    If any of the layers' parameters was set to a Theano expression instead
    of a shared variable, `unwrap_shared` controls whether to return the
    shared variables involved in that expression (``unwrap_shared=True``,
    the default), or the expression itself (``unwrap_shared=False``). In
    either case, tag filtering applies to the expressions, considering all
    variables within an expression to be tagged the same.

    Examples
    --------
    Collecting all parameters from a two-layer network:

    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)
    >>> l2 = DenseLayer(l1, num_units=30)
    >>> all_params = get_all_params(l2)
    >>> all_params == [l1.W, l1.b, l2.W, l2.b]
    True

    Parameters can be filtered by tags, and parameter expressions are
    unwrapped to return involved shared variables by default:

    >>> from lasagne.utils import floatX
    >>> w1 = theano.shared(floatX(.01 * np.random.randn(50, 30)))
    >>> w2 = theano.shared(floatX(1))
    >>> l2 = DenseLayer(l1, num_units=30, W=theano.tensor.exp(w1) - w2, b=None)
    >>> all_params = get_all_params(l2, regularizable=True)
    >>> all_params == [l1.W, w1, w2]
    True

    When disabling unwrapping, the expression for ``l2.W`` is returned instead:

    >>> all_params = get_all_params(l2, regularizable=True,
    ...                             unwrap_shared=False)
    >>> all_params == [l1.W, l2.W]
    True
    """
    layers = get_all_layers(layer)
    params = chain.from_iterable(l.get_params(
            unwrap_shared=unwrap_shared, **tags) for l in layers)
    return utils.unique(params)


def count_params(layer, **tags):
    pass


def get_all_param_values(layer, **tags):
    pass


def set_all_param_values(layer, values, **tags):
    pass
