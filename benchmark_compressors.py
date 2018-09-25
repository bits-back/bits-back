import io
import gzip
import bz2
import lzma
import numpy as np

from data.load_imagenet import load_imagenet_valid
from torchvision import datasets, transforms
import PIL.Image as pimg


def mnist_raw():
    mnist = datasets.MNIST(
        'data/mnist', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
    return mnist.test_data.numpy()

def mnist_binarized(rng):
    raw_probs = mnist_raw() / 255
    return rng.random_sample(np.shape(raw_probs)) < raw_probs

def bench_compressor(compress_fun, compressor_name, images, images_name):
    byts = compress_fun(images)
    n_bits = len(byts) * 8
    bits_per_pixel = n_bits / np.size(images)
    print("Dataset: {}. Compressor: {}. Rate: {:.2f} bits per channel.".
          format(images_name, compressor_name, bits_per_pixel))

def gzip_compress(images):
    original_size = np.size(images)
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype is np.dtype('uint8')
    return gzip.compress(images.tobytes())

def bz2_compress(images):
    original_size = np.size(images)
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype is np.dtype('uint8')
    return bz2.compress(images.tobytes())

def lzma_compress(images):
    original_size = np.size(images)
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype is np.dtype('uint8')
    return lzma.compress(images.tobytes())

def pimg_compress(format='PNG', **params):
    def compress_fun(images):
        compressed_data = bytearray()
        for n, image in enumerate(images):
            image = pimg.fromarray(image)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format=format, **params)
            compressed_data.extend(img_bytes.getvalue())
        return compressed_data
    return compress_fun

def gz_and_pimg(images, format='PNG', **params):
    pimg_compressed_data = pimg_compress(images, format, **params)
    return gzip.compress(pimg_compressed_data)


if __name__ == "__main__":
    # MNIST_raw
    images = mnist_raw()
    bench_compressor(gzip_compress, "gzip", images, 'raw mnist')
    bench_compressor(bz2_compress, "bz2", images, 'raw mnist')
    bench_compressor(lzma_compress, "lzma", images, 'raw mnist')
    bench_compressor(
        pimg_compress("PNG", optimize=True), "PNG", images, 'raw mnist')
    bench_compressor(
        pimg_compress('WebP', lossless=True, quality=100), "WebP", images, 'raw mnist')

    # MNIST binarized
    rng = np.random.RandomState(0)
    images = mnist_binarized(rng)
    bench_compressor(gzip_compress, "gzip", images, 'binarized mnist')
    bench_compressor(bz2_compress, "bz2", images, 'binarized mnist')
    bench_compressor(lzma_compress, "lzma", images, 'binarized mnist')
    bench_compressor(
        pimg_compress("PNG", optimize=True), "PNG", images, 'binarized mnist')
    bench_compressor(
        pimg_compress('WebP', lossless=True, quality=100), "WebP", images, 'binarized mnist')

    # ImageNet
    images = load_imagenet_valid()
    bench_compressor(gzip_compress, "gzip",  images, 'imagenet')
    bench_compressor(bz2_compress,   "bz2",  images, 'imagenet')
    bench_compressor(lzma_compress, "lzma",  images, 'imagenet')
    bench_compressor(
        pimg_compress("PNG", optimize=True), "PNG", images, 'binarized mnist')
    bench_compressor(
        pimg_compress('WebP', lossless=True, quality=100), "WebP", images,
        'imagenet')
