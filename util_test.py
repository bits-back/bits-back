from itertools import repeat

import numpy as np
import util
import rans

import numpy.testing as np_testing


rng = np.random.RandomState(0)


def check_cdf_inverse(cdf, ppf, input_prec, coder_prec):
    assert cdf(0) == 0
    assert all([ppf(cf) == s
                for s in range(1 << input_prec)
                for cf in range(cdf(s), cdf(s + 1))])
    assert cdf(1 << input_prec) == 1 << coder_prec

def test_gaussian_latent_cdf():
    mean, stdd = 0.1, 0.5
    prior_precision = 2
    post_precision = 12
    cdf = util.gaussian_latent_cdf(mean, stdd, prior_precision, post_precision)
    ppf = util.gaussian_latent_ppf(mean, stdd, prior_precision, post_precision)
    check_cdf_inverse(cdf, ppf, prior_precision, post_precision)

def test_beta_latent_cdf():
    a_prior, b_prior = 0.1, 0.5
    a_post, b_post = 5, 7
    prior_precision = 2
    post_precision = 5
    cdf = util.beta_latent_cdf(a_prior, b_prior, a_post, b_post,
                               prior_precision, post_precision)
    ppf = util.beta_latent_ppf(a_prior, b_prior, a_post, b_post,
                               prior_precision, post_precision)
    check_cdf_inverse(cdf, ppf, prior_precision, post_precision)

def test_binomial_cdf():
    data_precision = 8
    n = (1 << data_precision) - 1
    p = 0.3
    precision = 10
    check_cdf_inverse(util.binomial_cdf(n, p, precision),
                      util.binomial_ppf(n, p, precision),
                      data_precision, precision)

def check_append_pop(data, append, pop):
    state = rans.msg_init

    # Encode
    state = append(state, data)
    arr = rans.flatten(state)
    print("Message length: " + str(32 * len(arr)) + " bits.")

    # Decode
    state, data_reconstructed = pop(state)
    np_testing.assert_allclose(data, data_reconstructed)
    assert state == rans.msg_init

def test_uniform_append_pop():
    n_data = 100
    precision = 5
    data = rng.randint(1 << precision, size=n_data)
    check_append_pop(data, util.uniforms_append(precision),
                     util.uniforms_pop(precision, n_data))

def test_non_uniform_append_pop():
    n_data = 100
    precision = 5
    cfs = np.array([0, 3, 17, 19, 32])
    cdf = lambda s: cfs[s]
    ppf = lambda cf: np.searchsorted(cfs, cf, 'right') - 1
    data = rng.randint(1 << 2, size=n_data)
    check_append_pop(data,
                     util.non_uniforms_append(precision, repeat(cdf)),
                     util.non_uniforms_pop(precision, repeat(ppf, n_data),
                                           repeat(cdf, n_data)))

def test_categorical_cdf():
    coder_precision = 12
    prob = np.array([0.2, 0.3, 0.5, 0.6])
    s_precision = 2  # len(prob) == 1 << s_precision
    check_cdf_inverse(util.categorical_cdf(prob, coder_precision),
                      util.categorical_ppf(prob, coder_precision),
                      s_precision, coder_precision)

def test_categorical_append_pop():
    precision = 12
    categories = 6
    num_data = 1000

    probs = rng.randn(num_data, categories)
    probs = np.exp(probs)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    data = [rng.choice(categories, p=prob) for prob in probs]
    data = np.asarray(data)

    entropy = -np.sum(probs * np.log2(probs), axis=1)
    print('Optimal compression is: {:.2f} bits'.format(np.sum(entropy)))

    check_append_pop(data,
                     util.categoricals_append(probs, precision),
                     util.categoricals_pop(probs, precision))
