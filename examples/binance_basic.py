import asyncio
from pprint import pprint
from crypto_history import class_builders, init_logger, data_container_intra
import logging


async def main():
    init_logger(level=logging.INFO)
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
        reference_assets=["BTC", "USDT"],
        ohlcv_fields=desired_fields,
        time_range_dict=time_range
    )
    xdataarray_of_coins = await time_aggregated_data_container.get_time_aggregated_data_container()
    pprint(xdataarray_of_coins)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # TODO Identify where the unclosed session originates from
