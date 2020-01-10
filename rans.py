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


head_precision = 64
tail_precision = 32
tail_mask = (1 << tail_precision) - 1
head_min  = 1 << head_precision - tail_precision

#        head    , tail
x_init = head_min, ()

def append(x, start, freq, precision):
    """
    Encodes a symbol with range [`start`, `start + freq`).  All frequencies are
    assumed to sum to `1 << precision`, and compressed bits get written to x.
    """
    # Prevent Numpy scalars leaking in
    start, freq = int(start), int(freq)
    head, tail = x
    if head >= freq << head_precision - precision:
        # Need to push data down into tail
        head, tail = head >> tail_precision, (head & tail_mask, tail)
    return (head // freq << precision) + head % freq + start, tail

def pop(x, statfun, precision):
    """
    Pops a symbol from x. The signiature of statfun should be
        statfun: cf |-> symbol, (start, freq)
    where `cf` is in the interval [`start`, `freq`) and `symbol` is the symbol
    corresponding to that interval.
    """
    head, tail = x
    cf = head & ((1 << precision) - 1)
    symb, (start, freq) = statfun(cf)
    # Prevent Numpy scalars leaking in
    start, freq = int(start), int(freq)
    head = freq * (head >> precision) + cf - start
    if head < head_min:
        # Need to pull data up from tail
        head_new, tail = tail
        head = (head << tail_precision) + head_new
    return (head, tail), symb

def append_symbol(statfun, precision):
    def append_(x, symbol):
        start, freq = statfun(symbol)
        return append(x, start, freq, precision)
    return append_

def pop_symbol(statfun, precision):
    def pop_(x):
        return pop(x, statfun, precision)
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
