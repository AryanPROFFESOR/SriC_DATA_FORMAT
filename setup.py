"""Build both Cython extensions for sric v3."""
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

FLAGS = ["-O3", "-march=native", "-ffast-math",
         "-funroll-loops", "-fvisibility=hidden"]
DEFS  = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
INC   = [np.get_include()]

exts = cythonize([
    Extension("sric_ext._codec",
              sources=["sric_ext/_codec.pyx"],
              include_dirs=INC, extra_compile_args=FLAGS, define_macros=DEFS),
    Extension("sric_ext._zeromodel",
              sources=["sric_ext/_zeromodel.pyx"],
              include_dirs=INC, extra_compile_args=FLAGS, define_macros=DEFS),
], compiler_directives={
    "language_level": "3", "boundscheck": False, "wraparound": False,
    "cdivision": True, "nonecheck": False, "initializedcheck": False,
})

setup(
    name="sric", 
    version="3.0.0",
    packages=find_packages(exclude=["tests*", "build*"]),
    ext_modules=exts,
    install_requires=[
        "numpy>=1.23", 
        "scipy>=1.9",
        "requests",  # Required for the CLI fetch command
        "tqdm"       # Required for the CLI progress bar
    ],
    extras_require={
        "anndata": ["anndata>=0.9"], 
        "dev": ["pytest", "Cython>=3"]
    },
    entry_points={
        "console_scripts": ["sric=sric.cli:main"]
    },
)