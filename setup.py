from setuptools import dist
if __name__ == '__main__':
    dist.Distribution().fetch_build_eggs(['Cython>=0.29.1', 'numpy>=1.24.1', 'wheel>=0.37.1'])

from distutils.core import setup as csetup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


def get_extensions():
    extensions = [Extension('*', sources=['src/wwvec/raster_to_vector/color_grid.pyx']),
                  Extension('*', sources=['src/wwvec/raster_to_vector/thin_grid.pyx'])]
    return extensions


def setup_package():
    metadata = dict(
        ext_modules=cythonize(get_extensions(), annotate=True),
        include_dirs=[np.get_include()]
    )
    csetup(**metadata)

if __name__ == '__main__':
    setup_package()

