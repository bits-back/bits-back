import numpy as np
import torch
import rans
from benchmark_compressors import mnist_binarized, bz2_compress, pimg_compress

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.image import BboxImage

from torch_vae.torch_bin_mnist_compress import vae_append

rng = np.random.RandomState(0)

fig, ax = plt.subplots()

N = 20
spacing = 20
h = 28

raw_data = mnist_binarized(rng)[:N]

def bytes_to_img(byts):
    arr = np.unpackbits(np.frombuffer(byts, dtype='uint8'))
    arr = np.concatenate((arr, np.ones(28 - arr.size % 28)))
    return np.reshape(arr, (h, -1), order='F')

bz2_compressed = bytes_to_img(bz2_compress(raw_data))
png_compressed = bytes_to_img(pimg_compress(optimize=True)(raw_data))

other_bits = rng.randint(1 << 32, size=12, dtype=np.uint32)
state = rans.unflatten(other_bits)

images = torch.from_numpy(raw_data.astype('float32'))
images = [image.view(-1) for image in images]

for image in images:
    state = vae_append(state, image)
bb_ans_compressed = bytes_to_img(rans.flatten(state).tobytes())

raw_data = np.concatenate(~raw_data, axis=1)

labels = ["BB-ANS",          "bz2",          "PNG",          "MNIST"]
data   = [bb_ans_compressed, bz2_compressed, png_compressed, raw_data          ]

yticks = []
for i, d in enumerate(data):
    bbox = Bbox.from_bounds(0, 28 * (spacing + i * (h + spacing)),
                            28 * d.shape[1], 28 * h)
    bbox = TransformedBbox(bbox, ax.transData)
    bbox_image = BboxImage(bbox, cmap='gray', origin=None)
    bbox_image.set_data(d)
    ax.add_artist(bbox_image)
    yticks.append(28 * (spacing + i * (h + spacing) + h // 2))

ax.set_xlim(0, max(28 * d.shape[1] for d in data))
ax.set_yticks(yticks)
ax.set_yticklabels(labels)
ax.set_ylim(0, 28 * (spacing + (i + 1) * (h + spacing)))
ax.set_xlabel('Size (bits)')
ax.set_aspect('equal')
plt.savefig('compression_plot.png', dpi=800, bbox_inches='tight')
