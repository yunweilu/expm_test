"""
setup.py - a module to allow package installation
"""

from distutils.core import setup


NAME = "expm"
VERSION = "0.1"
DEPENDENCIES = [
    "numpy",
    "scipy",
]
DESCRIPTION = "a package for performing quantum optimal control"
AUTHOR = "Yunwei LU"
AUTHOR_EMAIL = "yunweilu2020@u.northwestern.edu"

setup(author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=DEPENDENCIES,
      name=NAME,
      version=VERSION,
)