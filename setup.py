#!/usr/bin/env python
import os

import setuptools
from setuptools import setup

setup(
    name="mae_factory",
    version="0.0.1",
    description="Modular vision transformer lbirary for rapid model prototyping",
    author="Matthew Fortier",
    author_email="fortier.matt@gmail.com",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "torch>=1.12.0",
        "Pillow",
        "torchvision",
    ],
    extras_require={},
)