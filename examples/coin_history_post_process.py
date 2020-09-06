import asyncio
from pprint import pprint
from crypto_history import class_builders, \
    init_logger, data_container_access, data_container_post
import logging


async def main():
    init_logger(level=logging.DEBUG)

    exchange_factory = class_builders.get("market").get("binance")()

    desired_fields = ["open_ts", "open", "close_ts"]
    time_aggregated_data_container = data_container_access.TimeAggregatedDataContainer(
        exchange_factory,
        base_assets=["NANO", "AMB", "XRP"],
        reference_assets=["BTC", "USDT"],
        ohlcv_fields=desired_fields,
        time_range_dict={("25 Jan 2018", "27 Feb 2018"): "1d",
                         ("26 Aug 2020", "now"):         "1w"}
    )
    xdataarray_of_coins = await time_aggregated_data_container.get_time_aggregated_data_container()
    pprint(xdataarray_of_coins)

    xdataset = xdataarray_of_coins.to_dataset("ohlcv_fields")
    pprint(xdataset)

    type_converter = data_container_post.TypeConvertedData(exchange_factory)
    type_converted_dataset = type_converter.set_type_on_dataset(xdataset)

    type_converted_dataarray = type_converter.set_type_on_dataarray(xdataarray_of_coins)

    incomplete_data_handle = data_container_post.HandleIncompleteData()
    entire_na_removed_dataarray = incomplete_data_handle.\
        drop_xarray_coins_with_entire_na(type_converted_dataarray)

    entire_na_removed_dataset = incomplete_data_handle.\
        drop_xarray_coins_with_entire_na(type_converted_dataset)

    strict_na_dropped_dataarray = incomplete_data_handle.\
        nullify_incomplete_data_from_dataarray(entire_na_removed_dataarray)
    pprint(strict_na_dropped_dataarray)

    strict_na_dropped_dataset = incomplete_data_handle.\
        nullify_incomplete_data_from_dataset(entire_na_removed_dataset)
    pprint(strict_na_dropped_dataset)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
