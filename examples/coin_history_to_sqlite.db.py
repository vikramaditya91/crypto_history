import asyncio
from crypto_history import class_builders, init_logger,\
    save_to_disk, data_container_post, data_container_access
import logging
import tempfile


async def main():
    init_logger(level=logging.DEBUG)

    exchange_factory = class_builders.get("market").get("binance")()

    desired_fields = ["open_ts", "open", "close"]
    time_aggregated_data_container = data_container_access.TimeAggregatedDataContainer(
        exchange_factory,
        base_assets=["NANO", "AMB", "XRP"],
        reference_assets=["BTC"],
        ohlcv_fields=desired_fields,
        time_range_dict={("25 Jan 2018", "27 Feb 2018"): "1d",
                         ("26 Aug 2020", "now"):         "1d"}
    )
    xdataarray_of_coins = await time_aggregated_data_container.get_time_aggregated_data_container()
    type_converter = data_container_post.TypeConvertedData(exchange_factory)
    type_converted_dataarray = type_converter.set_type_on_dataarray(xdataarray_of_coins)

    incomplete_data_handle = data_container_post.HandleIncompleteData()
    entire_na_removed_dataarray = incomplete_data_handle.\
        drop_xarray_coins_with_entire_na(type_converted_dataarray)
    strict_na_dropped_dataarray = incomplete_data_handle.\
        nullify_incomplete_data_from_dataarray(entire_na_removed_dataarray)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".db") as temp_file:
        sql_writer = class_builders.get("write_to_disk").get("sqlite")()
        save_to_disk.write_coin_history_to_file(strict_na_dropped_dataarray,
                                                sql_writer,
                                                temp_file.name,
                                                ["open", "close"])


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
