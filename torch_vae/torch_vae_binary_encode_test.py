import pickle

import torch
import numpy as np
import util
import rans
from torch_vae.tvae_binary import BinaryVAE
from torch_vae import tvae_utils

rng = np.random.RandomState(0)
prior_precision = 8
q_precision = 12
obs_precision = 12


def test_bvae_enc_dec():
    # load an mnist image, x_0
    image = pickle.load(open('torch_vae/sample_mnist_image', 'rb'))
    image = torch.round(torch.tensor(image)).float()

    # load vae params
    model = BinaryVAE()
    model.load_state_dict(
        torch.load('torch_vae/saved_params/torch_binary_vae_params'))

    latent_shape = (20,)

    rec_net = tvae_utils.torch_fun_to_numpy_fun(model.encode)
    gen_net = tvae_utils.torch_fun_to_numpy_fun(model.decode)

    obs_append = tvae_utils.bernoulli_obs_append(obs_precision)
    obs_pop = tvae_utils.bernoulli_obs_pop(obs_precision)

    vae_append = util.vae_append(
        latent_shape, gen_net, rec_net, obs_append,
        prior_precision, q_precision)

    vae_pop = util.vae_pop(
        latent_shape, gen_net, rec_net, obs_pop,
        prior_precision, q_precision)

    # randomly generate some 'other' bits
    other_bits = rng.randint(1 << 16, size=20, dtype=np.uint32)
    state = rans.msg_init
    state = util.uniforms_append(16)(state, other_bits)

    # ---------------------------- ENCODE ------------------------------------
    state = vae_append(state, image)
    compressed_message = rans.flatten(state)

    print("Used " + str(32 * (len(compressed_message) - len(other_bits))) +
          " bits.")

    # ---------------------------- DECODE ------------------------------------
    state = rans.unflatten(compressed_message)

    state, image_ = vae_pop(state)
    assert all(image == image_)

    #  recover the other bits from q(y|x_0)
    state, recovered_bits = util.uniforms_pop(16, 20)(state)
    assert all(other_bits == recovered_bits)
    assert state == rans.msg_init
