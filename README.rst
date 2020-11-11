================================
Welcome to crypto-history
================================


.. image:: https://img.shields.io/pypi/v/crypto-history.svg
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
    :target: https://coveralls.io/github/vikramaditya91/crypto_history?branch=master
    :alt: crypto-history coveralls coverage

.. image:: https://travis-ci.org/vikramaditya91/crypto_history.svg?branch=master
    :target: https://travis-ci.org/vikramaditya91/crypto_history


This is a wrapper on binance and other exchange APIs to aggregate historical information
in structured tabular formats (such as xarray.DataArray and SQLite).

Source code
  https://github.com/vikramaditya91/crypto_history

Documentation
  https://crypto-history.readthedocs.io/en/latest/

Features
--------

- Cleans the data ticker-wise if incomplete
- Sets the correct type on the data obtained
- Is able to join data from various chunks of time in a single DataArray
- Candles of varying intervals can be obtained in a single DataArray
- Fetches information about all tickers available on Binance asynchronously
- Delays requests if it is close to the limit prescribed by Binance
- Retries when the requests have exceeded the performance limit of the machine
- Obtains the history of each/all tickers in the xarray.DataArray format
- Easily extendable to other exchanges and other data formats
- It does not require an API key from Binance
- null values can be dropped either timestamp-wise and/or coin-wise
- Can export data in SQLite format and xr.DataArray
- Chunks of time can be aggregated into a single data object

Quick Start
-----------

.. code:: bash

    pip install crypto-history

See a basic example at :`examples/binance_basic.py <https://github.com/vikramaditya91/crypto_history/tree/master/examples/binance_basic.py>`_

.. code:: python

    exchange_factory = class_builders.get("market").get("binance")()

    desired_fields = ["open_ts", "open"]

    binance_homogenizer = exchange_factory.create_data_homogenizer()
    base_assets = await binance_homogenizer.get_all_base_assets()
    print(f"All the base assets available on the Binance exchange are {base_assets}")

    time_range = {("25 Jan 2020", "27 May 2020"): "1d",
                  ("26 Aug 2020", "now"):         "1h"}
    time_aggregated_data_container = data_container_intra.TimeAggregatedDataContainer(
        exchange_factory,
        base_assets=["NANO", "IOST", "XRP"],
        reference_assets=["BTC"],
        ohlcv_fields=desired_fields,
        time_range_dict=time_range
    )
    xdataarray_of_coins = await time_aggregated_data_container.get_time_aggregated_data_container()
    pprint(xdataarray_of_coins)


For more `check out the documentation <https://crypto-history.readthedocs.io/en/latest/>`_.




