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




