"""
setup.py
========
Build the C++ XPBD extension module.

    pip install pybind11 numpy
    python setup.py build_ext --inplace

After building, xpbd_core.cpython-*.so (Linux/Mac) or
xpbd_core.*.pyd (Windows) will appear in this directory.
wire_simulator.py will automatically use it.
"""

from setuptools import setup, Extension
import pybind11
import numpy as np

ext = Extension(
    "xpbd_core",
    sources=["xpbd_core.cpp"],
    include_dirs=[
        pybind11.get_include(),
        np.get_include(),
    ],
    extra_compile_args=[
        "/O2",          # MSVC optimisation (Windows)
        "/std:c++17",
    ],
    language="c++",
)

setup(
    name="xpbd_core",
    packages=[],        # no Python packages — just the C++ extension
    ext_modules=[ext],
)