"""
Cythonize
"""
import os, sys
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="Polya-Gamma Sampler",
    ext_modules=cythonize("./polya_gamma.pyx"),
    include_dirs=[numpy.get_include()],
    zip_false=False
)
