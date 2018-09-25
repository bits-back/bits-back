import numpy as np
import torch

import util
from torch_vae.tvae_beta_binomial import beta_binomial_log_pdf


def tensor_to_ndarray(tensor):
    if type(tensor) is tuple:
        return tuple(tensor_to_ndarray(t) for t in tensor)
    else:
        return tensor.detach().numpy()

def ndarray_to_tensor(arr):
    if type(arr) is tuple:
        return tuple(ndarray_to_tensor(a) for a in arr)
    elif type(arr) is torch.Tensor:
        return arr
    else:
        return torch.from_numpy(arr)

def torch_fun_to_numpy_fun(fun):
    def numpy_fun(*args, **kwargs):
        torch_args = ndarray_to_tensor(args)
        return tensor_to_ndarray(fun(*torch_args, **kwargs))
    return numpy_fun

def bernoulli_obs_append(precision):
    def obs_append(probs):
        def append(state, data):
            return util.bernoullis_append(probs, precision)(
                state, np.int64(data))
        return append
    return obs_append

def bernoulli_obs_pop(precision):
    def obs_pop(probs):
        def pop(state):
            state, data = util.bernoullis_pop(probs, precision)(state)
            return state, torch.Tensor(data)
        return pop
    return obs_pop

def beta_binomial_obs_append(n, precision):
    def obs_append(params):
        a, b = params
        def append(state, data):
            return util.beta_binomials_append(a, b, n, precision)(
                state, np.int64(data))
        return append
    return obs_append

def beta_binomial_obs_pop(n, precision):
    def obs_pop(params):
        a, b = params
        def pop(state):
            state, data = util.beta_binomials_pop(a, b, n, precision)(state)
            return state, torch.Tensor(data)
        return pop
    return obs_pop
