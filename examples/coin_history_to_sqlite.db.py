import asyncio
from crypto_history import class_builders, init_logger,\
    save_to_disk, data_container_post, data_container_access
import logging
import tempfile


async def main():
    init_logger(level=logging.DEBUG)

    exchange_factory = class_builders.get("market").get("binance")()

    async with exchange_factory.create_data_homogenizer() \
            as binance_homogenizer:
        base_assets = await binance_homogenizer.get_all_base_assets()

    desired_fields = ["open_ts", "open", "close"]
    candle_type = "1h"
    time_aggregated_data_container = data_container_access.TimeAggregatedDataContainer.create_instance(
        exchange_factory,
        base_assets=base_assets,
        reference_assets=["BTC"],
        ohlcv_fields=desired_fields,
        time_range_dict={("25 Jan 2017", "18 Nov 2020"): candle_type}
    )
    xdataarray_of_coins = await time_aggregated_data_container.get_time_aggregated_data_container()
    type_converter = data_container_post.TypeConvertedData(exchange_factory)
    type_converted_dataarray = type_converter.set_type_on_dataarray(xdataarray_of_coins)

    sql_writer = class_builders.get("write_to_disk").get("sqlite")()

    save_to_disk.write_coin_history_to_file(type_converted_dataarray,
                                            sql_writer,
                                            f"/home/vikramaditya/PycharmProjects/database/25_Jan_2017_TO_18_Nov_2020_BTC_{candle_type}1h.db",
                                            ["open", "close"])


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
