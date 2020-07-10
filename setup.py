from crypto_history.version import __version__
from setuptools import setup

PACKAGE_NAME = "crypto_history"
VERSION = __version__


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
        "various exchanges and presenting them in a data-format of choice"
    ),
    install_requires=[
        'python-binance @ git+ssh://git@github.com/sammchardy/python-binance.git#egg=python-binance-feature/asyncio',
        'xarray'
    ],
    keywords="binance cryptocurrency xarray",
    license="Simplified BSD License",
    version=VERSION,
)