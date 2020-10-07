import codecs
import os.path as op
from setuptools import setup

PACKAGE_NAME = "crypto_history"

here = op.abspath(op.dirname(__file__))


def read(rel_path):
    with codecs.open(op.join(here, rel_path), 'r') as fp:
        return fp.read()


with open(op.join(here, "README.rst"), encoding="utf-8") as fp:
    README = fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = "="
            return line.split(delim)[-1].strip().strip('"')
    else:
        raise RuntimeError("Unable to find version string.")


VERSION = get_version(op.join(op.dirname(__file__), PACKAGE_NAME, "version.py"))

extras = {
    "ci": ["coveralls"],
    "lint": ["black", "flake8", "pre-commit", "pydocstyle"],
    "test": [
        "pytest-vcr",
        "vcrpy==4.0.2",
        "mock >=0.8",
        "pytest",
    ],
}

extras["dev"] = extras["lint"] + extras["test"]

setup(
    name=PACKAGE_NAME,
    author="Vikramaditya Gaonkar",
    packages=["crypto_history",
              "crypto_history.data_container",
              "crypto_history.stock_market",
              "crypto_history.utilities"],
    url="https://github.com/vikramaditya91/crypto_history",
    author_email="vikramaditya91@gmail.com",
    python_requires=">=3.8.0",
    classifiers=[
        "Development Status :: 4 - Beta",
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
    extras_require=extras,
    install_requires=[
        'python-binance-async',
        'xarray',
        'python-dateutil'
    ],
    keywords="binance cryptocurrency xarray",
    license="Simplified BSD License",
    long_description=README,
    version=VERSION,
)
