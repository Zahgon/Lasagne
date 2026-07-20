import theano.tensor as T

from .. import init
from .. import nonlinearities
from ..utils import as_tuple, int_types, inspect_kwargs
from ..theano_extensions import conv

from .base import Layer


__all__ = [
    "Conv1DLayer",
    "Conv2DLayer",
    "Conv3DLayer",
    "TransposedConv2DLayer",
    "Deconv2DLayer",
    "DilatedConv2DLayer",
    "TransposedConv3DLayer",
    "Deconv3DLayer",
]


def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    input_length : int or None
        The size of the input.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

    Returns
    -------
    int or None
        The output size corresponding to the given convolution parameters, or
        ``None`` if `input_size` is ``None``.

    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.
    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int_types):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    output_length = (output_length + stride - 1) // stride

    return output_length


def conv_input_length(output_length, filter_size, stride, pad=0):
    """Helper function to compute the input size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    output_length : int or None
        The size of the output.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

    Returns
    -------
    int or None
        The smallest input size corresponding to the given convolution
        parameters for the given output size, or ``None`` if `output_size` is
        ``None``. For a strided convolution, any input size of up to
        ``stride - 1`` elements larger than returned will still give the same
        output size.

    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.

    Notes
    -----
    This can be used to compute the output size of a convolution backward pass,
    also called transposed convolution, fractionally-strided convolution or
    (wrongly) deconvolution in the literature.
    """
    if output_length is None:
        return None
    if pad == 'valid':
        pad = 0
    elif pad == 'full':
        pad = filter_size - 1
    elif pad == 'same':
        pad = filter_size // 2
    if not isinstance(pad, int_types):
        raise ValueError('Invalid pad: {0}'.format(pad))
    return (output_length - 1) * stride - 2 * pad + filter_size


class BaseConvLayer(Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 num_groups=1, n=None, **kwargs):
        super(BaseConvLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n+2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, n, int_types)
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, n, int_types)
        self.untie_biases = untie_biases

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, n, int_types)

        if (num_groups <= 0 or
                self.num_filters % num_groups != 0 or
                self.input_shape[1] % num_groups != 0):
            raise ValueError(
                "num_groups (here: %d) must be positive and evenly divide the "
                "number of input and output channels (here: %d and %d)" %
                (num_groups, self.input_shape[1], self.num_filters))
        elif (num_groups > 1 and
              "num_groups" not in inspect_kwargs(T.nnet.conv2d)):
            raise RuntimeError("num_groups > 1 requires "
                               "Theano 0.10 or later")  # pragma: no cover
        self.num_groups = num_groups

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1] // self.num_groups
        return (self.num_filters, num_input_channels) + self.filter_size

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)

    def convolve(self, input, **kwargs):
        """
        Symbolically convolves `input` with ``self.W``, producing an output of
        shape ``self.output_shape``. To be implemented by subclasses.

        Parameters
        ----------
        input : Theano tensor
            The input minibatch to convolve
        **kwargs
            Any additional keyword arguments from :meth:`get_output_for`

        Returns
        -------
        Theano tensor
            `input` convolved according to the configuration of this layer,
            without any bias or nonlinearity applied.
        """
        raise NotImplementedError("BaseConvLayer does not implement the "
                                  "convolve() method. You will want to "
                                  "use a subclass such as Conv2DLayer.")


class Conv1DLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=1,
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=conv.conv1d_mc0, **kwargs):
        super(Conv1DLayer, self).__init__(incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, flip_filters, n=1,
                                          **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        extra_kwargs = {}
        if self.num_groups > 1:  # pragma: no cover
            extra_kwargs['num_groups'] = self.num_groups
        conved = self.convolution(input, self.W,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters,
                                  **extra_kwargs)
        return conved


class Conv2DLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, flip_filters, n=2,
                                          **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        extra_kwargs = {}
        if self.num_groups > 1:  # pragma: no cover
            extra_kwargs['num_groups'] = self.num_groups
        conved = self.convolution(input, self.W,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters,
                                  **extra_kwargs)
        return conved


class Conv3DLayer(BaseConvLayer):  # pragma: no cover
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=None, **kwargs):
        super(Conv3DLayer, self).__init__(incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, flip_filters, n=3,
                                          **kwargs)
        if convolution is None:
            convolution = T.nnet.conv3d
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        extra_kwargs = {}
        if self.num_groups > 1:  # pragma: no cover
            extra_kwargs['num_groups'] = self.num_groups
        conved = self.convolution(input, self.W,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters,
                                  **extra_kwargs)
        return conved


if not hasattr(T.nnet, 'conv3d'):  # pragma: no cover
    del Conv3DLayer
    __all__.remove('Conv3DLayer')


class TransposedConv2DLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 crop=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=False,
                 output_size=None, **kwargs):
        if (not isinstance(output_size, T.Variable) and
                output_size is not None):
            output_size = as_tuple(output_size, 2, int_types)
        self.output_size = output_size
        super(TransposedConv2DLayer, self).__init__(
                incoming, num_filters, filter_size, stride, crop, untie_biases,
                W, b, nonlinearity, flip_filters, n=2, **kwargs)
        self.crop = self.pad
        del self.pad

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        if self.output_size is not None:
            size = self.output_size
            if isinstance(self.output_size, T.Variable):
                size = (None, None)
            return input_shape[0], self.num_filters, size[0], size[1]

        crop = getattr(self, 'crop', getattr(self, 'pad', None))
        crop = crop if isinstance(crop, tuple) else (crop,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_input_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, crop)))

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.crop == 'same' else self.crop
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=self.get_W_shape(),
            subsample=self.stride, border_mode=border_mode,
            filter_flip=not self.flip_filters)
        output_size = self.output_shape[2:]
        if isinstance(self.output_size, T.Variable):
            output_size = self.output_size
        elif any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]
        conved = op(self.W, input, output_size)
        return conved

Deconv2DLayer = TransposedConv2DLayer


class TransposedConv3DLayer(BaseConvLayer):  # pragma: no cover
    def __init__(self, incoming, num_filters, filter_size,
                 stride=(1, 1, 1), crop=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=False,
                 output_size=None, **kwargs):
        if (not isinstance(output_size, T.Variable) and
                output_size is not None):
            output_size = as_tuple(output_size, 3, int_types)
        self.output_size = output_size
        BaseConvLayer.__init__(self, incoming, num_filters, filter_size,
                               stride, crop, untie_biases, W, b,
                               nonlinearity, flip_filters, n=3, **kwargs)
        self.crop = self.pad
        del self.pad

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        if self.output_size is not None:
            size = self.output_size
            if isinstance(self.output_size, T.Variable):
                size = (None, None, None)
            return input_shape[0], self.num_filters, size[0], size[1], size[2]

        crop = getattr(self, 'crop', getattr(self, 'pad', None))
        crop = crop if isinstance(crop, tuple) else (crop,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_input_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, crop)))

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.crop == 'same' else self.crop
        op = T.nnet.abstract_conv.AbstractConv3d_gradInputs(
            imshp=self.output_shape,
            kshp=self.get_W_shape(),
            subsample=self.stride, border_mode=border_mode,
            filter_flip=not self.flip_filters)
        output_size = self.output_shape[2:]
        if isinstance(self.output_size, T.Variable):
            output_size = self.output_size
        elif any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]
        conved = op(self.W, input, output_size)
        return conved

Deconv3DLayer = TransposedConv3DLayer


if not hasattr(T.nnet.abstract_conv,
               'AbstractConv3d_gradInputs'):  # pragma: no cover
    del TransposedConv3DLayer, Deconv3DLayer
    __all__.remove('TransposedConv3DLayer')
    __all__.remove('Deconv3DLayer')


class DilatedConv2DLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, dilation=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=False,
                 **kwargs):
        self.dilation = as_tuple(dilation, 2, int_types)
        super(DilatedConv2DLayer, self).__init__(
                incoming, num_filters, filter_size, 1, pad,
                untie_biases, W, b, nonlinearity, flip_filters, n=2, **kwargs)
        del self.stride
        if self.pad != (0, 0):
            raise NotImplementedError(
                    "DilatedConv2DLayer requires pad=0 / (0,0) / 'valid', but "
                    "got %r. For a padded dilated convolution, add a PadLayer."
                    % (pad,))
        if self.flip_filters:
            raise NotImplementedError(
                    "DilatedConv2DLayer requires flip_filters=False.")

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, (filter-1) * dilate + 1, 1, 0)
                      for input, filter, dilate
                      in zip(input_shape[2:], self.filter_size,
                             self.dilation)))

    def convolve(self, input, **kwargs):
        imshp = self.input_shape
        kshp = self.output_shape
        imshp = (imshp[1], imshp[0]) + imshp[2:]
        kshp = (kshp[1], kshp[0]) + kshp[2:]
        op = T.nnet.abstract_conv.AbstractConv2d_gradWeights(
            imshp=imshp, kshp=kshp,
            subsample=self.dilation, border_mode='valid',
            filter_flip=False)
        output_size = self.output_shape[2:]
        if any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]
        conved = op(input.transpose(1, 0, 2, 3), self.W, output_size)
        return conved.transpose(1, 0, 2, 3)
