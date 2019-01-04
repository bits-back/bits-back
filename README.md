Code for the paper [Practical Lossless Compression with Latent Variables using Bits Back Coding](https://openreview.net/forum?id=ryE98iR5tm), appearing at ICLR 2019.

All of the code was written by [Jamie Townsend](https://github.com/j-towns/) and [Tom Bird](https://github.com/tom-bird).
### Overview of the code
The low level rANS encoding and decoding functions are in [rans.py](rans.py). Higher level functions for encoding and decoding according to various distributions, including using [BB-ANS coding](util.py#L152) and more specialised [BB-ANS VAE coding](util.py#L168), are in [util.py](util.py).

Scripts relating specifically to VAE learning and compression are in the [torch_vae](torch_vae) subdirectory. We have implemented two models, one for binarized MNIST digits and one for raw (non-binarized) digits. The script [torch_vae/torch_bin_mnist_compress.py](torch_vae/torch_bin_mnist_compress.py) compresses the binarized MNIST dataset using the learned VAE, [torch_vae/torch_mnist_compress.py](torch_vae/torch_mnist_compress.py) compresses raw MNIST. Pre-learned parameters for both models are in [torch_vae/saved_params](torch_vae/saved_params). The parameters were learned using the [torch_vae/tvae_binary.py](torch_vae/tvae_binary.py) and [torch_vae/tvae_beta_binomial.py](torch_vae/tvae_beta_binomial.py) scripts respectively.

The [benchmark_compressors.py](benchmark_compressors.py) script measures the compression rate of various commonly used lossless compression algorithms, including those quoted in our paper. In order to run the benchmarks for the Imagenet 64x64 dataset, you will need to download and unzip the dataset into the directory data/imagenet.

To run the tests, run [pytest](https://docs.pytest.org/en/latest/) from the root dir. Scripts in the `pytorch_vae` directory must be run from the root dir using
```
python -m torch_vae.[name_of_module]    # Without .py at the end.
```
### Known dependencies
Core: Python 3, Numpy, Scipy

VAE Compression: Pytorch

Compression benchmarks: [Pillow](https://pillow.readthedocs.io/en/stable/)

Please notify us if we've missed something.

### Acknowledgements
Thanks to [Jarek Duda](http://th.if.uj.edu.pl/~dudaj/) for inventing ANS and [fighting to keep it patent free](https://arstechnica.com/tech-policy/2018/06/inventor-says-google-is-patenting-work-he-put-in-the-public-domain/). Thanks to [Fabian Giesen](https://fgiesen.wordpress.com/) for the [ryg_rans](https://github.com/rygorous/ryg_rans) ANS implementation, upon which [rans.py](rans.py) is closely based.
