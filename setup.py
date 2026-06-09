from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "vrp_solver.inventory_fast",
        ["src/vrp_solver/inventory_fast.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
