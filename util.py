import numpy as np
from scipy.stats import norm, beta, binom
from scipy.special import gammaln
import rans


# ----------------------------------------------------------------------------
# Statistics functions for encoding and decoding according to uniform and non-
# uniform distributions over the integer symbols in range(1 << precision).
#
# An encoder statfun performs the mapping
#     symbol |--> (start, freq)
#
# A decoder statfun performs the mapping
#     cumulative_frequency |--> symbol, (start, freq)
# ----------------------------------------------------------------------------
uniform_enc_statfun = lambda s: (s, 1)
uniform_dec_statfun = lambda cf: (cf, (cf, 1))

def uniforms_append(precision):
    append_fun = rans.append_symbol(uniform_enc_statfun, precision)
    def append(state, symbols):
        for symbol in reversed(symbols):
            state = append_fun(state, symbol)
        return state
    return append

def uniforms_pop(precision, n):
    pop_fun = rans.pop_symbol(uniform_dec_statfun, precision)
    def pop(state):
        symbols = []
        for i in range(n):
            state, symbol = pop_fun(state)
            symbols.append(symbol)
        return state, np.asarray(symbols)
    return pop

def non_uniform_enc_statfun(cdf):
    def enc(s):
        start = cdf(s)
        freq = cdf(s + 1) - start
        return start, freq
    return enc

def non_uniform_dec_statfun(ppf, cdf):
    def dec(cf):
        idx = ppf(cf)
        start, freq = non_uniform_enc_statfun(cdf)(idx)
        assert start <= cf < start + freq
        return idx, (start, freq)
    return dec

def non_uniforms_append(precision, cdfs):
    def append(state, symbols):
        for symbol, cdf in reversed(list(zip(symbols, cdfs))):
            statfun = non_uniform_enc_statfun(cdf)
            state = rans.append_symbol(statfun, precision)(state, symbol)
        return state
    return append

def non_uniforms_pop(precision, ppfs, cdfs):
    def pop(state):
        symbols = []
        for ppf, cdf in zip(ppfs, cdfs):
            statfun = non_uniform_dec_statfun(ppf, cdf)
            state, symbol = rans.pop_symbol(statfun, precision)(state)
            symbols.append(symbol)
        return state, np.asarray(symbols)
    return pop

# ----------------------------------------------------------------------------
# Cumulative distribution functions and inverse cumulative distribution
# functions (ppf) for discretised Gaussian and Beta latent distributions.
#
# Latent cdf inputs are indices of buckets of equal width under the 'prior',
# assumed for the purposes of bits back to be in the same family. They lie in
# the range of ints [0, 1 << prior_prec)
#
# cdf outputs are scaled and rounded to map to integers in the range of ints
# [0, 1 << post_prec) instead of the range [0, 1]
#
# For decodability we must satisfy
#     all(ppf(cf) == s for s in range(1 << prior_prec) for cf in
#         range(cdf(s), cdf(s + 1)))
# ----------------------------------------------------------------------------
def _nearest_int(arr):
    # This will break when vectorized
    return int(np.around(arr))

std_gaussian_bucket_cache = {}  # Stores bucket endpoints
std_gaussian_centres_cache = {}  # Stores bucket centres

def std_gaussian_buckets(precision):
    """
    Return the endpoints of buckets partioning the domain of the prior. Each
    bucket has mass 1 / (1 << precision) under the prior.
    """
    if precision in std_gaussian_bucket_cache:
        return std_gaussian_bucket_cache[precision]
    else:
        buckets = np.float32(
            norm.ppf(np.arange((1 << precision) + 1) / (1 << precision)))
        std_gaussian_bucket_cache[precision] = buckets
        return buckets

def std_gaussian_centres(precision):
    """
    Return the centres of mass of buckets partioning the domain of the prior.
    Each bucket has mass 1 / (1 << precision) under the prior.
    """
    if precision in std_gaussian_centres_cache:
        return std_gaussian_centres_cache[precision]
    else:
        centres = np.float32(
            norm.ppf((np.arange(1 << precision) + 0.5) / (1 << precision)))
        std_gaussian_centres_cache[precision] = centres
        return centres

def gaussian_latent_cdf(mean, stdd, prior_prec, post_prec):
    def cdf(idx):
        x = std_gaussian_buckets(prior_prec)[idx]
        return _nearest_int(norm.cdf(x, mean, stdd) * (1 << post_prec))
    return cdf

def gaussian_latent_ppf(mean, stdd, prior_prec, post_prec):
    def ppf(cf):
        x = norm.ppf((cf + 0.5) / (1 << post_prec), mean, stdd)
        # Binary search is faster than using the actual gaussian cdf for the
        # precisions we typically use, however the cdf is O(1) whereas search
        # is O(precision), so for high precision cdf will be faster.
        return np.searchsorted(
            std_gaussian_buckets(prior_prec), x, 'right') - 1
    return ppf

def beta_latent_cdf(
        a_prior, b_prior, a_post, b_post, prior_prec, post_prec):
    def cdf(idx):
        x = beta.ppf(idx / (1 << prior_prec), a_prior, b_prior)
        return _nearest_int(beta.cdf(x, a_post, b_post) * (1 << post_prec))
    return cdf

def beta_latent_ppf(
        a_prior, b_prior, a_post, b_post, prior_prec, post_prec):
    def ppf(cf):
        x = beta.ppf((cf + 0.5) / (1 << post_prec), a_post, b_post)
        return (beta.cdf(x, a_prior, b_prior) * (1 << prior_prec)).astype(int)
    return ppf

# ----------------------------------------------------------------------------
# Bits back append and pop
# ----------------------------------------------------------------------------
def bb_ans_append(post_pop, lik_append, prior_append):
    def append(state, data):
        state, latent = post_pop(data)(state)
        state = lik_append(latent)(state, data)
        state = prior_append(state, latent)
        return state
    return append

def bb_ans_pop(prior_pop, lik_pop, post_append):
    def pop(state):
        state, latent = prior_pop(state)
        state, data = lik_pop(latent)(state)
        state = post_append(data)(state, latent)
        return state, data
    return pop

def vae_append(latent_shape, gen_net, rec_net, obs_append, prior_prec=8,
               latent_prec=12):
    """
    Assume that the vae uses an isotropic Gaussian for its prior and diagonal
    Gaussian for its posterior.
    """
    def post_pop(data):
        post_mean, post_stdd = rec_net(data)
        post_mean, post_stdd = np.ravel(post_mean), np.ravel(post_stdd)
        cdfs = [gaussian_latent_cdf(m, s, prior_prec, latent_prec)
                for m, s in zip(post_mean, post_stdd)]
        ppfs = [gaussian_latent_ppf(m, s, prior_prec, latent_prec)
                for m, s in zip(post_mean, post_stdd)]
        return non_uniforms_pop(latent_prec, ppfs, cdfs)

    def lik_append(latent_idxs):
        y = std_gaussian_centres(prior_prec)[latent_idxs]
        obs_params = gen_net(np.reshape(y, latent_shape))
        return obs_append(obs_params)

    prior_append = uniforms_append(prior_prec)
    return bb_ans_append(post_pop, lik_append, prior_append)

def vae_pop(
        latent_shape, gen_net, rec_net, obs_pop, prior_prec=8, latent_prec=12):
    """
    Assume that the vae uses an isotropic Gaussian for its prior and diagonal
    Gaussian for its posterior.
    """
    prior_pop = uniforms_pop(prior_prec, np.prod(latent_shape))

    def lik_pop(latent_idxs):
        y = std_gaussian_centres(prior_prec)[latent_idxs]
        obs_params = gen_net(np.reshape(y, latent_shape))
        return obs_pop(obs_params)

    def post_append(data):
        post_mean, post_stdd = rec_net(data)
        post_mean, post_stdd = np.ravel(post_mean), np.ravel(post_stdd)
        cdfs = [gaussian_latent_cdf(m, s, prior_prec, latent_prec)
                for m, s in zip(post_mean, post_stdd)]
        return non_uniforms_append(latent_prec, cdfs)

    return bb_ans_pop(prior_pop, lik_pop, post_append)

# ----------------------------------------------------------------------------
# Functions for Bernoulli and categorical distributions
# ----------------------------------------------------------------------------
def create_categorical_buckets(probs, precision):
    buckets = np.rint(probs * ((1 << precision) - len(probs))) + np.ones(probs.shape)
    bucket_sum = sum(buckets)
    if not bucket_sum == 1 << precision:
        i = np.argmax(buckets)
        buckets[i] += (1 << precision) - bucket_sum
    assert sum(buckets) == 1 << precision
    return np.insert(np.cumsum(buckets), 0, 0)  # this could be slightly wrong

def categorical_cdf(probs, precision):
    def cdf(s):
        cumulative_buckets = create_categorical_buckets(probs, precision)
        return int(cumulative_buckets[s])
    return cdf

def categorical_ppf(probs, precision):
    def ppf(cf):
        cumulative_buckets = create_categorical_buckets(probs, precision)
        return np.searchsorted(cumulative_buckets, cf, 'right') - 1
    return ppf

def categoricals_append(probs, precision):
    """Assume that the last dim of probs contains the probability vectors,
    i.e. np.sum(probs, axis=-1) == ones"""
    # Flatten all but last dim of probs
    probs = np.reshape(probs, (-1, np.shape(probs)[-1]))
    cdfs = [categorical_cdf(p, precision) for p in probs]
    def append(state, data):
        data = np.ravel(data)
        return non_uniforms_append(precision, cdfs)(state, data)
    return append

def categoricals_pop(probs, precision):
    """Assume that the last dim of probs contains the probability vectors,
    i.e. np.sum(probs, axis=-1) == ones"""
    # Flatten all but last dim of probs
    data_shape = np.shape(probs)[:-1]
    probs = np.reshape(probs, (-1, np.shape(probs)[-1]))
    cdfs = [categorical_cdf(p, precision) for p in probs]
    ppfs = [categorical_ppf(p, precision) for p in probs]

    def pop(state):
        state, symbols = non_uniforms_pop(precision, ppfs, cdfs)(state)
        return state, np.reshape(symbols, data_shape)
    return pop

def bernoullis_append(probs, precision):
    return categoricals_append(np.stack((1 - probs, probs), axis=-1), precision)

def bernoullis_pop(probs, precision):
    return categoricals_pop(np.stack((1 - probs, probs), axis=-1), precision)

def binomial_cdf(n, p, precision):
    def cdf(k):
        return _nearest_int(binom.cdf(k - 1, n, p) * (1 << precision))
    return cdf

def binomial_ppf(n, p, precision):
    def ppf(cf):
        return np.int64(binom.ppf((cf + 0.5) / (1 << precision), n, p))
    return ppf

def beta_binomial_log_pdf(k, n, a, b):
    a_plus_b = a + b
    numer = (gammaln(n + 1) + gammaln(k + a) + gammaln(n - k + b)
             + gammaln(a_plus_b))
    denom = (gammaln(k + 1) + gammaln(n - k + 1) + gammaln(n + a_plus_b)
             + gammaln(a) + gammaln(b))
    return numer - denom

def generate_beta_binomial_probs(a, b, n):
    ks = np.arange(n + 1)
    a = a[..., np.newaxis]
    b = b[..., np.newaxis]
    probs = np.exp(beta_binomial_log_pdf(ks, n, a, b))
    # make sure normalised, there are some numerical
    # issues with the exponentiation in the beta binomial
    probs = np.clip(probs, 1e-10, 1.)
    return probs / np.sum(probs, axis=-1, keepdims=True)

def beta_binomials_append(a, b, n, precision):
    # TODO: Implement this using bits-back instead of generic discrete distrn.
    probs = generate_beta_binomial_probs(a, b, n)
    return categoricals_append(probs, precision)

def beta_binomials_pop(a, b, n, precision):
    # TODO: Implement this using bits-back instead of generic discrete distrn.
    probs = generate_beta_binomial_probs(a, b, n)
    return categoricals_pop(probs, precision)
