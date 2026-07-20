import theano.tensor as T
from .layers import Layer, get_all_params


def l1(x):
    pass


def l2(x):
    pass


def apply_penalty(tensor_or_tensors, penalty, **kwargs):
    pass


def regularize_layer_params(layer, penalty,
                            tags={'regularizable': True}, **kwargs):
    pass


def regularize_layer_params_weighted(layers, penalty,
                                     tags={'regularizable': True}, **kwargs):
    pass


def regularize_network_params(layer, penalty,
                              tags={'regularizable': True}, **kwargs):
    pass
