import asyncio
import logging
import pathlib

from crypto_history import class_builders, init_logger, \
    save_to_disk, data_container_post, data_container_access


async def main():
    init_logger(level=logging.DEBUG)

    exchange_factory = class_builders.get("market").get("binance")()

    async with exchange_factory.create_data_homogenizer() \
            as binance_homogenizer:
        base_assets = await binance_homogenizer.get_all_base_assets()

    desired_fields = ["open_ts", "open", "close"]
    candle_type = "1m"

    time_aggregated_data_container = data_container_access.TimeAggregatedDataContainer.create_instance(
        exchange_factory,
        base_assets=base_assets,
        reference_assets=["BTC"],
        ohlcv_fields=desired_fields,
        time_range_dict={("25 Jan 2018", "25 Jan 2021"): candle_type}
    )
    type_converter = data_container_post.TypeConvertedData(exchange_factory)
    type_conversion = type_converter.set_type_on_dataarray

    sql_writer = class_builders.get("write_to_disk").get("sqlite_pieces")()

    output_path = pathlib.Path(pathlib.Path(__file__).resolve()).parents[4] / \
                  "s3_sync" / \
                  f"25_Jan_2018_TO_18_Nov_2021_BTC_{candle_type}_directory"

    await save_to_disk.write_coin_history_to_file(sql_writer,
                                                  time_aggregated_data_container,
                                                  output_path=output_path,
                                                  operations={"post": [type_conversion],
                                                              "fields": ["open", "close"]})


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
