from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = 'My gradient module',
	ext_modules = cythonize("gd2c.pyx"),
	)