import theano
import theano.tensor as T
import numpy as np

from .. import init
from .. import nonlinearities
from ..utils import as_tuple, floatX, int_types
from ..random import get_rng
from .base import Layer, MergeLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


__all__ = [
    "NonlinearityLayer",
    "BiasLayer",
    "ScaleLayer",
    "standardize",
    "ExpressionLayer",
    "InverseLayer",
    "TransformerLayer",
    "TPSTransformerLayer",
    "ParametricRectifierLayer",
    "prelu",
    "RandomizedRectifierLayer",
    "rrelu",
]


class NonlinearityLayer(Layer):
    def __init__(self, incoming, nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(NonlinearityLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

    def get_output_for(self, input, **kwargs):
        return self.nonlinearity(input)


class BiasLayer(Layer):
    def __init__(self, incoming, b=init.Constant(0), shared_axes='auto',
                 **kwargs):
        super(BiasLayer, self).__init__(incoming, **kwargs)

        if shared_axes == 'auto':
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int_types):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes

        if b is None:
            self.b = None
        else:
            shape = [size for axis, size in enumerate(self.input_shape)
                     if axis not in self.shared_axes]
            if any(size is None for size in shape):
                raise ValueError("BiasLayer needs specified input sizes for "
                                 "all axes that biases are not shared over.")
            self.b = self.add_param(b, shape, 'b', regularizable=False)

    def get_output_for(self, input, **kwargs):
        if self.b is not None:
            bias_axes = iter(range(self.b.ndim))
            pattern = ['x' if input_axis in self.shared_axes
                       else next(bias_axes)
                       for input_axis in range(input.ndim)]
            return input + self.b.dimshuffle(*pattern)
        else:
            return input


class ScaleLayer(Layer):
    def __init__(self, incoming, scales=init.Constant(1), shared_axes='auto',
                 **kwargs):
        super(ScaleLayer, self).__init__(incoming, **kwargs)

        if shared_axes == 'auto':
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int_types):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes

        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ScaleLayer needs specified input sizes for "
                             "all axes that scales are not shared over.")
        self.scales = self.add_param(
            scales, shape, 'scales', regularizable=False)

    def get_output_for(self, input, **kwargs):
        axes = iter(range(self.scales.ndim))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes) for input_axis in range(input.ndim)]
        return input * self.scales.dimshuffle(*pattern)


def standardize(layer, offset, scale, shared_axes='auto'):
    """
    Convenience function for standardizing inputs by applying a fixed offset
    and scale.  This is usually useful when you want the input to your network
    to, say, have zero mean and unit standard deviation over the feature
    dimensions.  This layer allows you to include the appropriate statistics to
    achieve this normalization as part of your network, and applies them to its
    input.  The statistics are supplied as the `offset` and `scale` parameters,
    which are applied to the input by subtracting `offset` and dividing by
    `scale`, sharing dimensions as specified by the `shared_axes` argument.

    Parameters
    ----------
    layer : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    offset : Theano shared variable or numpy array
        The offset to apply (via subtraction) to the axis/axes being
        standardized.
    scale : Theano shared variable or numpy array
        The scale to apply (via division) to the axis/axes being standardized.
    shared_axes : 'auto', int or tuple of int
        The axis or axes to share the offset and scale over. If ``'auto'`` (the
        default), share over all axes except for the second: this will share
        scales over the minibatch dimension for dense layers, and additionally
        over all spatial dimensions for convolutional layers.

    Examples
    --------
    Assuming your training data exists in a 2D numpy ndarray called
    ``training_data``, you can use this function to scale input features to the
    [0, 1] range based on the training set statistics like so:

    >>> import lasagne
    >>> import numpy as np
    >>> training_data = np.random.standard_normal((100, 20))
    >>> input_shape = (None, training_data.shape[1])
    >>> l_in = lasagne.layers.InputLayer(input_shape)
    >>> offset = training_data.min(axis=0)
    >>> scale = training_data.max(axis=0) - training_data.min(axis=0)
    >>> l_std = standardize(l_in, offset, scale, shared_axes=0)

    Alternatively, to z-score your inputs based on training set statistics, you
    could set ``offset = training_data.mean(axis=0)`` and
    ``scale = training_data.std(axis=0)`` instead.
    """
    layer = BiasLayer(layer, -offset, shared_axes)
    layer.params[layer.b].remove('trainable')
    layer = ScaleLayer(layer, floatX(1.)/scale, shared_axes)
    layer.params[layer.scales].remove('trainable')
    return layer


class ExpressionLayer(Layer):
    def __init__(self, incoming, function, output_shape=None, **kwargs):
        super(ExpressionLayer, self).__init__(incoming, **kwargs)

        if output_shape is None:
            self._output_shape = None
        elif output_shape == 'auto':
            self._output_shape = 'auto'
        elif hasattr(output_shape, '__call__'):
            self.get_output_shape_for = output_shape
        else:
            self._output_shape = tuple(output_shape)

        self.function = function

    def get_output_shape_for(self, input_shape):
        if self._output_shape is None:
            return input_shape
        elif self._output_shape is 'auto':
            input_shape = (0 if s is None else s for s in input_shape)
            X = theano.tensor.alloc(0, *input_shape)
            output_shape = self.function(X).shape.eval()
            output_shape = tuple(s if s else None for s in output_shape)
            return output_shape
        else:
            return self._output_shape

    def get_output_for(self, input, **kwargs):
        return self.function(input)


class InverseLayer(MergeLayer):
    def __init__(self, incoming, layer, **kwargs):

        super(InverseLayer, self).__init__(
            [incoming, layer, layer.input_layer], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[2]

    def get_output_for(self, inputs, **kwargs):
        input, layer_out, layer_in = inputs
        return theano.grad(None, wrt=layer_in, known_grads={layer_out: input})


class TransformerLayer(MergeLayer):
    def __init__(self, incoming, localization_network, downsample_factor=1,
                 border_mode='nearest', **kwargs):
        super(TransformerLayer, self).__init__(
            [incoming, localization_network], **kwargs)
        self.downsample_factor = as_tuple(downsample_factor, 2)
        self.border_mode = border_mode

        input_shp, loc_shp = self.input_shapes

        if loc_shp[-1] != 6 or len(loc_shp) != 2:
            raise ValueError("The localization network must have "
                             "output shape: (batch_size, 6)")
        if len(input_shp) != 4:
            raise ValueError("The input network must have a 4-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns)")

    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        factors = self.downsample_factor
        return (shape[:2] + tuple(None if s is None else int(s // f)
                                  for s, f in zip(shape[2:], factors)))

    def get_output_for(self, inputs, **kwargs):
        input, theta = inputs
        return _transform_affine(theta, input, self.downsample_factor,
                                 self.border_mode)


def _transform_affine(theta, input, downsample_factor, border_mode):
    num_batch, num_channels, height, width = input.shape
    theta = T.reshape(theta, (-1, 2, 3))

    out_height = T.cast(height // downsample_factor[0], 'int64')
    out_width = T.cast(width // downsample_factor[1], 'int64')
    grid = _meshgrid(out_height, out_width)

    T_g = T.dot(theta, grid)
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
        input_dim, x_s_flat, y_s_flat,
        out_height, out_width, border_mode)

    output = T.reshape(
        input_transformed, (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output


def _interpolate(im, x, y, out_height, out_width, border_mode):
    num_batch, height, width, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1

    if border_mode == 'nearest':
        x0 = T.clip(x0_f, 0, width_f - 1)
        x1 = T.clip(x1_f, 0, width_f - 1)
        y0 = T.clip(y0_f, 0, height_f - 1)
        y1 = T.clip(y1_f, 0, height_f - 1)
    elif border_mode == 'mirror':
        w = 2 * (width_f - 1)
        x0 = T.minimum(x0_f % w, -x0_f % w)
        x1 = T.minimum(x1_f % w, -x1_f % w)
        h = 2 * (height_f - 1)
        y0 = T.minimum(y0_f % h, -y0_f % h)
        y1 = T.minimum(y1_f % h, -y1_f % h)
    elif border_mode == 'wrap':
        x0 = T.mod(x0_f, width_f)
        x1 = T.mod(x1_f, width_f)
        y0 = T.mod(y0_f, height_f)
        y1 = T.mod(y1_f, height_f)
    else:
        raise ValueError("border_mode must be one of "
                         "'nearest', 'mirror', 'wrap'")
    x0, x1, y0, y1 = (T.cast(v, 'int64') for v in (x0, x1, y0, y1))

    dim2 = width
    dim1 = width*height
    base = T.repeat(
        T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
    return output


def _linspace(start, stop, num):
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


def _meshgrid(height, width):
    x_t = T.dot(T.ones((height, 1)),
                _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid


class TPSTransformerLayer(MergeLayer):

    def __init__(self, incoming, localization_network, downsample_factor=1,
                 control_points=16, precompute_grid='auto',
                 border_mode='nearest', **kwargs):
        super(TPSTransformerLayer, self).__init__(
                [incoming, localization_network], **kwargs)

        self.border_mode = border_mode
        self.downsample_factor = as_tuple(downsample_factor, 2)
        self.control_points = control_points

        input_shp, loc_shp = self.input_shapes

        if loc_shp[-1] != 2 * control_points or len(loc_shp) != 2:
            raise ValueError("The localization network must have "
                             "output shape: (batch_size, "
                             "2*control_points)")

        if round(np.sqrt(control_points)) != np.sqrt(
                control_points):
            raise ValueError("The number of control points must be"
                             " a perfect square.")

        if len(input_shp) != 4:
            raise ValueError("The input network must have a 4-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns)")

        can_precompute_grid = all(s is not None for s in input_shp[2:])
        if precompute_grid == 'auto':
            precompute_grid = can_precompute_grid
        elif precompute_grid and not can_precompute_grid:
            raise ValueError("Grid can only be precomputed if the input "
                             "height and width are pre-specified.")
        self.precompute_grid = precompute_grid

        self.right_mat, self.L_inv, self.source_points, self.out_height, \
            self.out_width = _initialize_tps(
                control_points, input_shp, self.downsample_factor,
                precompute_grid)

    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        factors = self.downsample_factor
        return (shape[:2] + tuple(None if s is None else int(s // f)
                                  for s, f in zip(shape[2:], factors)))

    def get_output_for(self, inputs, **kwargs):
        input, dest_offsets = inputs
        return _transform_thin_plate_spline(
                dest_offsets, input, self.right_mat, self.L_inv,
                self.source_points, self.out_height, self.out_width,
                self.precompute_grid, self.downsample_factor, self.border_mode)


def _transform_thin_plate_spline(
        dest_offsets, input, right_mat, L_inv, source_points, out_height,
        out_width, precompute_grid, downsample_factor, border_mode):

    num_batch, num_channels, height, width = input.shape
    num_control_points = source_points.shape[1]

    dest_points = source_points + T.reshape(
            dest_offsets, (num_batch, 2, num_control_points))

    coefficients = T.dot(dest_points, L_inv[:, 3:].T)

    if precompute_grid:

        right_mat = T.tile(right_mat.dimshuffle('x', 0, 1), (num_batch, 1, 1))
        transformed_points = T.batched_dot(coefficients, right_mat)

    else:

        out_height = T.cast(height // downsample_factor[0], 'int64')
        out_width = T.cast(width // downsample_factor[1], 'int64')
        orig_grid = _meshgrid(out_height, out_width)
        orig_grid = orig_grid[0:2, :]
        orig_grid = T.tile(orig_grid, (num_batch, 1, 1))

        transformed_points = _get_transformed_points_tps(
                orig_grid, source_points, coefficients, num_control_points,
                num_batch)

    x_transformed = transformed_points[:, 0].flatten()
    y_transformed = transformed_points[:, 1].flatten()

    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
            input_dim, x_transformed, y_transformed,
            out_height, out_width, border_mode)

    output = T.reshape(input_transformed,
                       (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output


def _get_transformed_points_tps(new_points, source_points, coefficients,
                                num_points, batch_size):
    """
    Calculates the transformed points' value using the provided coefficients

    :param new_points: num_batch x 2 x num_to_transform tensor
    :param source_points: 2 x num_points array of source points
    :param coefficients: coefficients (should be shape (num_batch, 2,
        control_points + 3))
    :param num_points: the number of points

    :return: the x and y coordinates of each transformed point. Shape (
        num_batch, 2, num_to_transform)
    """


    to_transform = new_points.dimshuffle(0, 'x', 1, 2)
    stacked_transform = T.tile(to_transform, (1, num_points, 1, 1))
    r_2 = T.sum(((stacked_transform - source_points.dimshuffle(
            'x', 1, 0, 'x')) ** 2), axis=2)

    log_r_2 = T.log(r_2)
    distances = T.switch(T.isnan(log_r_2), r_2 * log_r_2, 0.)

    upper_array = T.concatenate([T.ones((batch_size, 1, new_points.shape[2]),
                                        dtype=theano.config.floatX),
                                 new_points], axis=1)
    right_mat = T.concatenate([upper_array, distances], axis=1)

    new_value = T.batched_dot(coefficients, right_mat)
    return new_value


def _U_func_numpy(x1, y1, x2, y2):
    pass


def _initialize_tps(num_control_points, input_shape, downsample_factor,
                    precompute_grid):
    pass


class ParametricRectifierLayer(Layer):
    def __init__(self, incoming, alpha=init.Constant(0.25), shared_axes='auto',
                 **kwargs):
        super(ParametricRectifierLayer, self).__init__(incoming, **kwargs)
        if shared_axes == 'auto':
            self.shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif shared_axes == 'all':
            self.shared_axes = tuple(range(len(self.input_shape)))
        elif isinstance(shared_axes, int_types):
            self.shared_axes = (shared_axes,)
        else:
            self.shared_axes = shared_axes

        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ParametricRectifierLayer needs input sizes for "
                             "all axes that alpha's are not shared over.")
        self.alpha = self.add_param(alpha, shape, name="alpha",
                                    regularizable=False)

    def get_output_for(self, input, **kwargs):
        axes = iter(range(self.alpha.ndim))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes)
                   for input_axis in range(input.ndim)]
        alpha = self.alpha.dimshuffle(pattern)
        return theano.tensor.nnet.relu(input, alpha)


def prelu(layer, **kwargs):
    pass


class RandomizedRectifierLayer(Layer):
    def __init__(self, incoming, lower=0.3, upper=0.8, shared_axes='auto',
                 **kwargs):
        super(RandomizedRectifierLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.lower = lower
        self.upper = upper

        if not isinstance(lower > upper, theano.Variable) and lower > upper:
            raise ValueError("Upper bound for RandomizedRectifierLayer needs "
                             "to be higher than lower bound.")

        if shared_axes == 'auto':
            self.shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif shared_axes == 'all':
            self.shared_axes = tuple(range(len(self.input_shape)))
        elif isinstance(shared_axes, int_types):
            self.shared_axes = (shared_axes,)
        else:
            self.shared_axes = shared_axes

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true, the arithmetic mean of lower and upper are used for the
            leaky slope.
        """
        if deterministic or self.upper == self.lower:
            return theano.tensor.nnet.relu(input, (self.upper+self.lower)/2.0)
        else:
            shape = list(self.input_shape)
            if any(s is None for s in shape):
                shape = list(input.shape)
            for ax in self.shared_axes:
                shape[ax] = 1

            rnd = self._srng.uniform(tuple(shape),
                                     low=self.lower,
                                     high=self.upper,
                                     dtype=theano.config.floatX)
            rnd = theano.tensor.addbroadcast(rnd, *self.shared_axes)
            return theano.tensor.nnet.relu(input, rnd)


def rrelu(layer, **kwargs):
    pass
