import asyncio
from pprint import pprint
from dataclasses import replace
from crypto_history import class_builders, init_logger, data_container
import logging
from binance import client

async def main():
    init_logger(level=logging.DEBUG)
    exchange_factory = class_builders.get("market").get("binance")()

    desired_fields = ["open_ts", "open", "high", "low", "close", "close_ts"]
    desired_fields = ["open_ts", "open"]

    binance_homogenizer = exchange_factory.create_data_homogenizer()
    base_assets = await binance_homogenizer.get_all_base_assets()
    base_assets = base_assets[::]
    base_assets = ["NANO", "IOST", "XRP"]

    time_range = {("25 Jan 2020", "27 May 2020"): "1h",
                  ("27 May 2020", "27 Aug 2020"): "1d",
                  ("26 Aug 2020", "now"):         "1m"}
    time_aggregated_data_container = data_container.TimeAggregatedDataContainer(exchange_factory,
                                                                                base_assets=base_assets,
                                                                                reference_assets=["BTC"],
                                                                                ohlcv_fields=desired_fields,
                                                                                time_range_dict=time_range,
                                                                                )
    xdataset_of_coins = await time_aggregated_data_container.get_time_aggregated_data_container()
    pprint(xdataset_of_coins)

    # data_container_factory = class_builders.get("data").get("xdataset")()
    # coin_history_obtainer = data_container_factory.create_coin_history_obtainer(exchange_factory,
    #                                                                             interval="1d",
    #                                                                             start_str="25 May 2020",
    #                                                                             end_str="27 May 2020",
    #                                                                             limit=1000
    #                                                                             )
    # container_dimensions_manager = data_container_factory.\
    #     create_data_container_dimensions_manager(coin_history_obtainer)
    # data_operations = data_container_factory.create_data_container_operations(coin_history_obtainer,
    #                                                                           container_dimensions_manager)
    # all_coordinates = await container_dimensions_manager.get_mapped_coords()
    # print(f"All coordinates available: {all_coordinates}")
    #
    # original_data_container = await data_operations.get_populated_original_container(
    #     ohlcv_fields=desired_fields,
    #     base_assets=all_coordinates.base_assets,
    #     reference_assets=all_coordinates.reference_assets
    # )
    # pprint(original_data_container)
    # index_manipulator = data_container_factory.create_data_container_index_manipulator(original_data_container)
    # modified_data_container = index_manipulator.get_timestamp_indexed_container()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # TODO Identify where the unclosed session originates from
