import asyncio
from pprint import pprint
from crypto_history import class_builders, \
    init_logger, data_container_access
import logging


async def main():
    init_logger(level=logging.DEBUG)

    exchange_factory = class_builders.get("market").get("binance")()

    desired_fields = ["open_ts", "open", "close_ts"]
    async with exchange_factory.create_data_homogenizer() \
            as binance_homogenizer:
        base_assets = await binance_homogenizer.get_all_base_assets()
        print(f"All the base assets available on the "
              f"Binance exchange are {base_assets}")

        reference_assets = await binance_homogenizer.get_all_refernce_assets()
        print(f"All the reference assets available on the"
              f" Binance exchange are {reference_assets}")

    time_range = {("25 Jan 2018", "27 Feb 2018"): "1d"}
    time_aggregated_data_container = data_container_access.\
        TimeAggregatedDataContainer(
            exchange_factory,
            base_assets=["NANO"],
            reference_assets=["BTC", "USDT"],
            ohlcv_fields=desired_fields,
            time_range_dict=time_range
        )
    xdataarray_of_coins = await time_aggregated_data_container.get_time_aggregated_data_container()
    pprint(xdataarray_of_coins)

    xdataset = xdataarray_of_coins.to_dataset("ohlcv_fields")
    pprint(xdataset)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
