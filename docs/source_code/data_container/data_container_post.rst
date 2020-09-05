Post Processing Data
===============================

The following provide information on how the historical data obtained in the form
of xarray.DataArray as described here can be post-processed.

You can find an example on post-processing at :`examples/coin_history_post_process.py <https://github.com/vikramaditya91/crypto_history/tree/master/examples/coin_history_post_process.py>`_

The currently offered post-processing capabilities are:

Type Conversion
-----------------

The original data obtained from the exchange may or may not be set with the correct type.
An example of this is Binance which provides the `open` (the opening value of the ticker)
as a string. The type converter stores the same value as a float.

The types are stored in :class:`the OHLCVFields <crypto_history.stock_market.stock_market_factory.AbstractOHLCVFieldTypes.OHLCVFields>`


.. autoclass:: crypto_history.data_container.data_container_post.TypeConvertedData()
    :members:
    :undoc-members:


Incomplete Data Deletion
--------------------------

Incomplete data from the xarray.DataArray or xarray.DataSet may have to be removed
to avoid unexpected behaviour and to save memory. It offers removal of incomplete data
in two ways.
If all the data corresponding to a particular base or reference asset is not available,
it can remove that coin from the xarray item.
If one of the values corresponding to a particular ticker is nan, it can make the entire
ticker contents nan.

.. autoclass:: crypto_history.data_container.data_container_post.HandleIncompleteData()
    :members:
    :undoc-members:
