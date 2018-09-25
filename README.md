To run the tests, checkout this repo and run [pytest](https://docs.pytest.org/en/latest/) from the root dir. Scripts in the `pytorch_vae` directory must be run from the root dir using
```
python -m torch_vae.[name_of_module]    # Without .py at the end.
```
__Known dependencies__ (please notify us if we've missed something):

Core: Python 3, Numpy, Scipy

VAE Compression: Pytorch

Compression benchmarks: [Pillow](https://pillow.readthedocs.io/en/stable/)
