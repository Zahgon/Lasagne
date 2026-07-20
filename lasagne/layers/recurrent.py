# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
from .. import nonlinearities
from .. import init
from ..utils import unroll_scan

from .base import MergeLayer, Layer
from .input import InputLayer
from .dense import DenseLayer
from . import helper

__all__ = [
    "CustomRecurrentLayer",
    "RecurrentLayer",
    "Gate",
    "LSTMLayer",
    "GRULayer"
]


class CustomRecurrentLayer(MergeLayer):
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        super(CustomRecurrentLayer, self).__init__(incomings, **kwargs)

        input_to_hidden_in_layers = \
            [layer for layer in helper.get_all_layers(input_to_hidden)
             if isinstance(layer, InputLayer)]
        if len(input_to_hidden_in_layers) != 1:
            raise ValueError(
                '`input_to_hidden` must have exactly one InputLayer, but it '
                'has {}'.format(len(input_to_hidden_in_layers)))

        hidden_to_hidden_in_lyrs = \
            [layer for layer in helper.get_all_layers(hidden_to_hidden)
             if isinstance(layer, InputLayer)]
        if len(hidden_to_hidden_in_lyrs) != 1:
            raise ValueError(
                '`hidden_to_hidden` must have exactly one InputLayer, but it '
                'has {}'.format(len(hidden_to_hidden_in_lyrs)))
        hidden_to_hidden_in_layer = hidden_to_hidden_in_lyrs[0]

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        if (self.precompute_input and
                input_to_hidden.output_shape[0] is not None and
                input_shape[0] is not None and
                input_shape[1] is not None and
                (input_to_hidden.output_shape[0] !=
                 input_shape[0]*input_shape[1])):
            raise ValueError(
                'When precompute_input == True, '
                'input_to_hidden.output_shape[0] must equal '
                'incoming.output_shape[0]*incoming.output_shape[1] '
                '(i.e. batch_size*sequence_length) or be None but '
                'input_to_hidden.output_shape[0] = {} and '
                'incoming.output_shape[0]*incoming.output_shape[1] = '
                '{}'.format(input_to_hidden.output_shape[0],
                            input_shape[0]*input_shape[1]))

        if (not self.precompute_input and
                input_to_hidden.output_shape[0] is not None and
                hidden_to_hidden.output_shape[0] is not None and
                (input_to_hidden.output_shape[0] !=
                 hidden_to_hidden.output_shape[0])):
            raise ValueError(
                'When precompute_input == False, '
                'input_to_hidden.output_shape[0] must equal '
                'hidden_to_hidden.output_shape[0] but '
                'input_to_hidden.output_shape[0] = {} and '
                'hidden_to_hidden.output_shape[0] = {}'.format(
                    input_to_hidden.output_shape[0],
                    hidden_to_hidden.output_shape[0]))

        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(input_to_hidden.output_shape[1:],
                                     hidden_to_hidden.output_shape[1:])):
            raise ValueError("The output shape for input_to_hidden and "
                             "hidden_to_hidden must be equal after the first "
                             "dimension, but input_to_hidden.output_shape={} "
                             "and hidden_to_hidden.output_shape={}".format(
                                 input_to_hidden.output_shape,
                                 hidden_to_hidden.output_shape))

        h_to_h_input_shape = hidden_to_hidden_in_layer.output_shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(input_to_hidden.output_shape[1:],
                                     h_to_h_input_shape[1:])):
            raise ValueError(
                "The output shape for input_to_hidden must be equal to the "
                "input shape of hidden_to_hidden after the first dimension, "
                "but input_to_hidden.output_shape={} and "
                "hidden_to_hidden:input_layer.shape={}".format(
                    input_to_hidden.output_shape, h_to_h_input_shape))

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1,) + hidden_to_hidden.output_shape[1:],
                name="hid_init", trainable=learn_init, regularizable=False)

    def get_params(self, **tags):
        params = super(CustomRecurrentLayer, self).get_params(**tags)
        params += helper.get_all_params(self.input_to_hidden, **tags)
        params += helper.get_all_params(self.hidden_to_hidden, **tags)
        return params

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_final:
            return (input_shape[0],) + self.hidden_to_hidden.output_shape[1:]
        else:
            return ((input_shape[0], input_shape[1]) +
                    self.hidden_to_hidden.output_shape[1:])

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        input = inputs[0]
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        input = input.dimshuffle(1, 0, *range(2, input.ndim))
        seq_len, num_batch = input.shape[0], input.shape[1]

        if self.precompute_input:
            trailing_dims = tuple(input.shape[n] for n in range(2, input.ndim))
            input = T.reshape(input, (seq_len*num_batch,) + trailing_dims)
            input = helper.get_output(
                self.input_to_hidden, input, **kwargs)

            trailing_dims = tuple(input.shape[n] for n in range(1, input.ndim))
            input = T.reshape(input, (seq_len, num_batch) + trailing_dims)

        non_seqs = helper.get_all_params(self.hidden_to_hidden)
        if not self.precompute_input:
            non_seqs += helper.get_all_params(self.input_to_hidden)

        def step(input_n, hid_previous, *args):
            hid_pre = helper.get_output(
                self.hidden_to_hidden, hid_previous, **kwargs)

            if self.precompute_input:
                hid_pre += input_n
            else:
                hid_pre += helper.get_output(
                    self.input_to_hidden, input_n, **kwargs)

            if self.grad_clipping:
                hid_pre = theano.gradient.grad_clip(
                    hid_pre, -self.grad_clipping, self.grad_clipping)

            return self.nonlinearity(hid_pre)


        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            dot_dims = (list(range(1, self.hid_init.ndim - 1)) +
                        [0, self.hid_init.ndim - 1])
            hid_init = T.dot(T.ones((num_batch, 1)),
                             self.hid_init.dimshuffle(dot_dims))

        if self.unroll_scan:
            input_shape = self.input_shapes[0]
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, *range(2, hid_out.ndim))

            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class RecurrentLayer(CustomRecurrentLayer):
    def __init__(self, incoming, num_units,
                 W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        if isinstance(incoming, tuple):
            input_shape = incoming
        else:
            input_shape = incoming.output_shape
        if 'name' in kwargs:
            basename = kwargs['name'] + '.'
            layer_kwargs = dict((key, arg) for key, arg in kwargs.items()
                                if key != 'name')
        else:
            basename = ''
            layer_kwargs = kwargs
        in_to_hid = DenseLayer(InputLayer((None,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None,
                               name=basename + 'input_to_hidden',
                               **layer_kwargs)
        hid_to_hid = DenseLayer(InputLayer((None, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None,
                                name=basename + 'hidden_to_hidden',
                                **layer_kwargs)

        self.W_in_to_hid = in_to_hid.W
        self.W_hid_to_hid = hid_to_hid.W
        self.b = in_to_hid.b

        super(RecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping, unroll_scan=unroll_scan,
            precompute_input=precompute_input, mask_input=mask_input,
            only_return_final=only_return_final, **kwargs)


class Gate(object):
    def __init__(self, W_in=init.Normal(0.1), W_hid=init.Normal(0.1),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_in = W_in
        self.W_hid = W_hid
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity


class LSTMLayer(MergeLayer):
    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        super(LSTMLayer, self).__init__(incomings, **kwargs)

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            pass

        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_final:
            return input_shape[0], self.num_units
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        input = inputs[0]
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            input = T.dot(input, W_in_stacked) + b_stacked

        def slice_w(x, n):
            s = x[:, n*self.num_units:(n+1)*self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]


        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked]
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            input_shape = self.input_shapes[0]
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, 2)

            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class GRULayer(MergeLayer):
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        super(GRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            pass

        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_final:
            return input_shape[0], self.num_units
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        input = inputs[0]
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        if self.precompute_input:
            input = T.dot(input, W_in_stacked) + b_stacked

        def slice_w(x, n):
            s = x[:, n*self.num_units:(n+1)*self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        def step(input_n, hid_previous, *args):
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            hid = (1 - updategate)*hid_previous + updategate*hidden_update
            return hid


        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        non_seqs = [W_hid_stacked]
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            input_shape = self.input_shapes[0]
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, 2)

            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out
