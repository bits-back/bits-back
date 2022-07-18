"""
Closely based on https://github.com/rygorous/ryg_rans/blob/master/rans64.h by
Fabian Giesen.

We use the pythonic names `append` and `pop` for encoding and decoding
respectively. The compressed state is a pair `msg = (head, tail)`, where `head`
is an int in the range `[0, 2 ** head_precision)` and `tail` is an immutable
stack, implemented using a cons list, containing ints in the range
`[0, 2 ** tail_precision)`. The precisions must satisfy

  tail_precision < head_precision <= 2 * tail_precision.

For convenient compatibility with Numpy dtypes we use the settings
head_precision = 64 and tail_precision = 32.

Both the `append` method and the `pop` method assume access to a probability
distribution over symbols. We use the name `symb` for a symbol. To describe the
probability distribution we model the real interval [0, 1] with the range of
integers {0, 1, 2, ..., 2 ** precision}. Each symbol is represented by a
sub-interval within that range. This can be visualized for a probability
distribution over the set of symbols {a, b, c, d}:

    0                                                             1
    |          |----- P(symb) ------|                             |
    |                                                             |
    |    a           symb == b           c              d         |
    |----------|--------------------|---------|-------------------|
    |                                                             |
    |          |------ prob --------|                             |
    0        start                                            2 ** precision

Each sub-interval can be represented by a pair of non-negative integers:
`start` and `prob`. As shown in the above diagram, the number `prob` represents
the width of the interval, corresponding to `symb`, so that

  P(symb) = prob / 2 ** precision

where P is the probability mass function of our distribution.

The number `start` represents the beginning of the interval corresponding to
`symb`, which is analagous to the cumulative distribution function evaluated on
`symb`.
"""
import numpy as np
from functools import reduce


head_precision = 64
tail_precision = 32
tail_mask = (1 << tail_precision) - 1
head_min  = 1 << head_precision - tail_precision

#          head    , tail
msg_init = head_min, ()

def append(msg, start, prob, precision):
    """
    Encodes a symbol with range `[start, start + prob)`.  All `prob`s are
    assumed to sum to `2 ** precision`. Compressed bits get written to `msg`.
    """
    # Prevent Numpy scalars leaking in
    start, prob, precision = map(int, [start, prob, precision])
    head, tail = msg
    if head >= prob << head_precision - precision:
        # Need to push data down into tail
        head, tail = head >> tail_precision, (head & tail_mask, tail)
    return (head // prob << precision) + head % prob + start, tail

def pop(msg, statfun, precision):
    """
    Pops a symbol from msg. The signiature of statfun should be
        statfun: cf |-> symb, (start, prob)
    where `cf` is in the interval `[start, start + prob)` and `symb` is the
    symbol corresponding to that interval.
    """
    # Prevent Numpy scalars leaking in
    precision = int(precision)
    head, tail = msg
    cf = head & ((1 << precision) - 1)
    symb, (start, prob) = statfun(cf)
    # Prevent Numpy scalars leaking in
    start, prob = int(start), int(prob)
    head = prob * (head >> precision) + cf - start
    if head < head_min:
        # Need to pull data up from tail
        head_new, tail = tail
        head = (head << tail_precision) + head_new
    return (head, tail), symb

def append_symbol(statfun, precision):
    def append_(msg, symbol):
        start, prob = statfun(symbol)
        return append(msg, start, prob, precision)
    return append_

def pop_symbol(statfun, precision):
    def pop_(msg):
        return pop(msg, statfun, precision)
    return pop_

def flatten(msg):
    """Flatten a rANS message into a 1d numpy array."""
    # We perform the bit-mask below to avoid an overflow error on systems which
    # use 32-bit Python builtin ints, see
    # https://github.com/numpy/numpy/issues/6289.
    out, msg = [msg[0] >> 32, msg[0] & tail_mask], msg[1]
    while msg:
        x_head, msg = msg
        out.append(x_head)
    return np.asarray(out, dtype=np.uint32)

def unflatten(arr):
    """Unflatten a 1d numpy array into a rANS message."""
    return (int(arr[0]) << 32 | int(arr[1]),
            reduce(lambda tl, hd: (int(hd), tl), reversed(arr[2:]), ()))
