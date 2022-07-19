from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

triangle_hash_module = Extension(
    'triangle_hash',
    sources=[
        'triangle_hash.pyx'
    ],
    include_dirs=[numpy.get_include()],
    libraries=['m']  # Unix-like specific
)

setup(ext_modules=cythonize([triangle_hash_module]))