from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="LDDBM",
    version="0.0.1",
    description="Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge",
    long_description=long_description,
    url="",
    python_requires=">=3.8",
    # Only add packages which are not available in the Bosch conda channels.
    install_requires=[],
    packages=find_packages(),
)
