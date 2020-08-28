================================
Welcome to crypto-history
================================


.. image:: https://img.shields.io/pypi/v/crypto-history.svg
    :target: https://pypi.python.org/pypi/crypto-history

.. image:: https://img.shields.io/pypi/l/crypto-history.svg
    :target: https://pypi.python.org/pypi/crypto-history

.. image:: https://img.shields.io/pypi/wheel/crypto-history.svg
    :target: https://pypi.python.org/pypi/crypto-history

.. image:: https://img.shields.io/pypi/pyversions/crypto-history.svg
    :target: https://pypi.python.org/pypi/crypto-history

.. image:: https://readthedocs.org/projects/crypto-history/badge/?version=latest
    :target: https://crypto-history.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://pepy.tech/badge/crypto-history/week
    :target: https://pepy.tech/badge/crypto-history/week
    :alt: crypto-history download status per week

.. image:: https://coveralls.io/repos/github/vikramaditya91/crypto_history/badge.svg?branch=feature/match-index-dataframe
    :target: https://coveralls.io/github/vikramaditya91/crypto_history?branch=feature/match-index-dataframe
    :alt: crypto-history coveralls coverage


This is a wrapper on binance and other exchange APIs to aggregate historical information
in structured tabular formats (such as xarray.DataArray)

Source code
  https://github.com/vikramaditya91/crypto-history

Documentation
  https://crypto-history.readthedocs.io/en/latest/

Features
--------

- Fetches information about all tickers available on Binance asynchronously
- Delays requests if it is close to the limit prescribed by Binance
- Retries when the requests have exceeded the performance limit of the machine
- Obtains the history of each/all tickers in the xarray.DataArray format
- Easily extendable to other exchanges and other data formats
- It does not require an API key from Binance

Quick Start
-----------

.. code:: bash

    pip install crypto-history

See `examples/binance_basic.py <https://github.com/vikramaditya/crypto_history/examples/binance_basic.py>`_ for a working example

.. code:: python

    exchange_factory = class_builders.get("market").get("binance")()
    data_container_factory = class_builders.get("data").get("xarray")()

    coin_history_obtainer = await data_container_factory.create_coin_history_obtainer(exchange_factory,
                                                                                      interval="1d",
                                                                                      start_str="1 January 2020",
                                                                                      end_str="4 June 2020",
                                                                                      limit=1000
                                                                                      )
    data_operations = await data_container_factory.create_data_container_operations(coin_history_obtainer)
    data_container = await data_operations.get_filled_container()
    pprint(data_container)


For more `check out the documentation <https://crypto-history.readthedocs.io/en/latest/>`_.




