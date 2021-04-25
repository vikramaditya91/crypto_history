Changelog
=========

1.2b12 (2021-Apr-25)
--------------------
 * bugfix/refix issue #20 vectorize empty numpy array

1.2b11 (2021-Apr-25)
-------------------
 * bugfix/fixes issue #20 vectorize with empty numpy array

1.2b10 (2021-Apr-5)
-------------------
 * feature/change build package to poetry

1.2b9 (2020-Nov-11)
-------------------
 * bugfix/write SQL even with incomplete data

1.2b7 (2020-Nov-09)
------------------
 * write to sqlite database coin history


1.2b6 (2020-Oct-25)
------------------
 * fixed aiohttp version to 3.7.1

1.2b5 (2020-Oct-24)
------------------
 * time-range history can be created with timedelta

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
