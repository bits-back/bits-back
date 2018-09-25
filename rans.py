"""
Closely based on https://github.com/rygorous/ryg_rans/blob/master/rans64.h

ROUGH GUIDE:
We use the pythonic names 'append' and 'pop' for encoding and decoding
respectively. The compressed state 'x' is an immutable stack, implemented using
a cons list.

x: the current stack-like state of the encoder/decoder.

precision: the natural numbers are divided into ranges of size 2^precision.

start & freq: start indicates the beginning of the range in [0, 2^precision-1]
that the current symbol is represented by. freq is the length of the range.
freq is chosen such that p(symbol) ~= freq/2^precision.
"""
import numpy as np
from functools import reduce


rans_l = 1 << 31  # the lower bound of the normalisation interval
tail_bits = (1 << 32) - 1

x_init = (rans_l, ())

def append(x, start, freq, precision):
    """Encodes a symbol with range [start, start + freq).  All frequencies are
    assumed to sum to "1 << precision", and the resulting bits get written to
    x."""
    if x[0] >= ((rans_l >> precision) << 32) * freq:
        x = (x[0] >> 32, (x[0] & tail_bits, x[1]))
    return ((x[0] // freq) << precision) + (x[0] % freq) + start, x[1]

def pop(x_, precision):
    """Advances in the bit stream by "popping" a single symbol with range start
    "start" and frequency "freq"."""
    cf = x_[0] & ((1 << precision) - 1)
    def pop(start, freq):
        x = freq * (x_[0] >> precision) + cf - start, x_[1]
        return ((x[0] << 32) | x[1][0], x[1][1]) if x[0] < rans_l else x
    return cf, pop

def append_symbol(statfun, precision):
    def append_(x, symbol):
        start, freq = statfun(symbol)
        return append(x, start, freq, precision)
    return append_

def pop_symbol(statfun, precision):
    def pop_(x):
        cf, pop_fun = pop(x, precision)
        symbol, (start, freq) = statfun(cf)
        return pop_fun(start, freq), symbol
    return pop_

def flatten(x):
    """Flatten a rans state x into a 1d numpy array."""
    out, x = [x[0] >> 32, x[0]], x[1]
    while x:
        x_head, x = x
        out.append(x_head)
    return np.asarray(out, dtype=np.uint32)

def unflatten(arr):
    """Unflatten a 1d numpy array into a rans state."""
    return (int(arr[0]) << 32 | int(arr[1]),
            reduce(lambda tl, hd: (int(hd), tl), reversed(arr[2:]), ()))
