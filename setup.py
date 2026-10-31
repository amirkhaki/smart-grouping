from setuptools import find_packages, setup

setup(
    name="smart-grouping",
    version="0.1.0",
    description="Latent Slice Classification Benchmarking on Frozen ELIC",
    packages=find_packages(),
    py_modules=["Network", "train", "Inference", "updata"],
)
