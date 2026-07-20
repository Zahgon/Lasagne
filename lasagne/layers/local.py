import theano.tensor as T

from .. import init
from .. import nonlinearities

from .conv import Conv2DLayer


__all__ = [
    "LocallyConnected2DLayer",
]


class LocallyConnected2DLayer(Conv2DLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad='same', untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 channelwise=False, **kwargs):
        self.channelwise = channelwise
        super(LocallyConnected2DLayer, self).__init__(
            incoming, num_filters, filter_size, stride=stride, pad=pad,
            untie_biases=untie_biases, W=W, b=b, nonlinearity=nonlinearity,
            flip_filters=flip_filters, **kwargs)
        if self.stride != (1, 1):
            raise NotImplementedError(
                "LocallyConnected2DLayer requires stride=1 / (1, 1), but got "
                "%r." % (stride,))
        if self.pad != 'same':
            raise NotImplementedError(
                "LocallyConnected2DLayer requires pad='same', but got %r." %
                (pad,))

    def get_W_shape(self):
        if any(s is None for s in self.input_shape[1:]):
            raise ValueError(
                "A LocallyConnected2DLayer requires a fixed input shape "
                "(except for the batch size). Got %r." % (self.input_shape,))
        num_input_channels = self.input_shape[1]
        output_shape = self.get_output_shape_for(self.input_shape)
        if self.channelwise:
            if self.channelwise and self.num_filters != num_input_channels:
                raise ValueError("num_filters and the number of input "
                                 "channels should match when channelwise is "
                                 "true, but got num_filters=%r and %d input "
                                 "channels" %
                                 (self.num_filters, num_input_channels))
            return (self.num_filters,) + self.filter_size + output_shape[-2:]
        else:
            return (self.num_filters, num_input_channels) + \
                   self.filter_size + output_shape[-2:]

    def convolve(self, input, **kwargs):
        output_shape = self.output_shape

        i = self.filter_size[0] // 2
        j = self.filter_size[1] // 2
        filter_h_ind = -i-1 if self.flip_filters else i
        filter_w_ind = -j-1 if self.flip_filters else j
        if self.channelwise:
            conved = input * self.W[:, filter_h_ind, filter_w_ind, :, :]
        else:
            conved = \
                (input[:, None, :, :, :] *
                 self.W[:, :, filter_h_ind, filter_w_ind, :, :]).sum(axis=-3)

        for i in range(self.filter_size[0]):
            filter_h_ind = -i-1 if self.flip_filters else i
            ii = i - (self.filter_size[0] // 2)
            input_h_slice = slice(
                max(ii, 0), min(ii + output_shape[-2], output_shape[-2]))
            output_h_slice = slice(
                max(-ii, 0), min(-ii + output_shape[-2], output_shape[-2]))

            for j in range(self.filter_size[1]):
                filter_w_ind = -j-1 if self.flip_filters else j
                jj = j - (self.filter_size[1] // 2)
                input_w_slice = slice(
                    max(jj, 0), min(jj + output_shape[-1], output_shape[-1]))
                output_w_slice = slice(
                    max(-jj, 0), min(-jj + output_shape[-1], output_shape[-1]))
                if ii == jj == 0:
                    continue
                if self.channelwise:
                    inc = (input[:, :, input_h_slice, input_w_slice] *
                           self.W[:, filter_h_ind, filter_w_ind,
                                  output_h_slice, output_w_slice])
                else:
                    inc = (input[:, None, :, input_h_slice, input_w_slice] *
                           self.W[:, :, filter_h_ind, filter_w_ind,
                                  output_h_slice, output_w_slice]).sum(axis=-3)
                conved = T.inc_subtensor(
                    conved[:, :, output_h_slice, output_w_slice], inc)
        return conved
