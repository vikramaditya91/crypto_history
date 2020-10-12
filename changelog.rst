Changelog
=========

1.2b4 (2020-Oct-12)
------------------
 * weighted data containers available
 * ability to change string weights to float


1.2b3 (2020-Oct-08)
------------------
 * build python package with python files included

1.2b1 (2020-Sep-06)
------------------
 * partially or fully incomplete histories can be purged if necessary
 * their types are set according to the ohlcv-field
 * Auto-build readthedocs on pull-request enabled

1.2b0 (2020-Aug-30)
------------------
 * timestamp chunks are generated based on the max-limit provided by the exchange
 * time histories are concatenated in a single dataarray

1.1a3 (2020-Aug-28)
------------------
 * xr.DataArray obtained indexed by the timestamp of user's choice
 * timestamp chunks are generated based on the max-limit provided by the exchange

1.1a2 (2020-Aug-25)
------------------

 * Allows the user to selectively choose which coordinates are to be pulled
 * Obtains the coin-history from Binance and allows it to be used on Binance
 * pytest framework and tox tests included
