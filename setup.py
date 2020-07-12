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
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Utilities",
    ],
    description=(
        "crypto_history is a python package for extracting history of crypto-currencies from "
        "various exchanges and presenting them ivn a data-format of choice"
    ),
    dependency_links=[
        "https://github.com/vikramaditya91/python-binance/tarball/feature/asyncio-release#egg=python-binance-0.7.3.b1"
    ],
    install_requires=[
        'python-binance',
        'xarray',
    ],
    keywords="binance cryptocurrency xarray",
    license="Simplified BSD License",
    version=VERSION,
)