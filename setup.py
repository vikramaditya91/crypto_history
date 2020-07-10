import codecs
import os.path as op
from setuptools import setup

PACKAGE_NAME = "crypto_history"


def read(rel_path):
    here = op.abspath(op.dirname(__file__))
    with codecs.open(op.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = "="
            return line.split(delim)[-1]
    else:
        raise RuntimeError("Unable to find version string.")


VERSION = get_version(op.join(op.dirname(__file__), PACKAGE_NAME, "version.py"))


setup(
    name=PACKAGE_NAME,
    author="Vikramaditya Gaonkar",
    author_email="vikramaditya91@gmail.com",
    python_requires=">3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Debug",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Utilities",
    ],
    description=(
        "crypto_history is a python package for extracting history of crypto-currencies from "
        "various exchanges and presenting them ivn a data-format of choice"
    ),
    install_requires=[
        "markdown",
        'python-binance @ git+ssh://git@github.com/sammchardy/python-binance.git#egg=python-binance-feature/asyncio',
        'xarray'
    ],
    keywords="binance cryptocurrency xarray",
    license="Simplified BSD License",
    version=VERSION,
)