import rans
import numpy as np


rng = np.random.RandomState(0)


def test_rans():
    x = rans.msg_init
    scale_bits = 8
    starts = rng.randint(0, 256, size=1000)
    freqs = rng.randint(1, 256, size=1000) % (256 - starts)
    freqs[freqs == 0] = 1
    assert np.all(starts + freqs <= 256)
    print("Exact entropy: " + str(np.sum(np.log2(256 / freqs))) + " bits.")
    # Encode
    for start, freq in zip(starts, freqs):
        x = rans.append(x, start, freq, scale_bits)
    coded_arr = rans.flatten(x)
    assert coded_arr.dtype == np.uint32
    print("Actual output size: " + str(32 * len(coded_arr)) + " bits.")

    # Decode
    x = rans.unflatten(coded_arr)
    for start, freq in reversed(list(zip(starts, freqs))):
        def statfun(cf):
            assert start <= cf < start + freq
            return None, (start, freq)
        x, symbol = rans.pop(x, statfun, scale_bits)
    assert x == (rans.head_min, ())


def test_flatten_unflatten():
    state = rans.msg_init
    some_bits = rng.randint(1 << 8, size=5)
    for b in some_bits:
        state = rans.append(state, b, 1, 8)
    flat = rans.flatten(state)
    state_ = rans.unflatten(flat)
    flat_ = rans.flatten(state_)
    assert np.all(flat == flat_)
    assert state == state_
