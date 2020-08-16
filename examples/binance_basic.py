import asyncio
from pprint import pprint
from dataclasses import replace
from crypto_history import class_builders, init_logger
import logging


async def main():
    init_logger(level=logging.DEBUG)
    exchange_factory = class_builders.get("market").get("binance")()
    data_container_factory = class_builders.get("data").get("xarray")()

    coin_history_obtainer = data_container_factory.create_coin_history_obtainer(exchange_factory,
                                                                                interval="1d",
                                                                                start_str="25 May 2020",
                                                                                end_str="27 May 2020",
                                                                                limit=1000
                                                                                )
    container_dimensions_manager = data_container_factory.\
        create_data_container_dimensions_manager(coin_history_obtainer)
    all_mapped_coords = await container_dimensions_manager.get_mapped_coords()
    coords_necessary = replace(all_mapped_coords,
                               field=['open_ts', 'open', 'close_ts'])
    data_operations = data_container_factory.create_data_container_operations(coin_history_obtainer,
                                                                              container_dimensions_manager)
    original_data_container = await data_operations.get_populated_original_container(coords_necessary)
    pprint(original_data_container)
    index_manipulator = data_container_factory.create_data_container_index_manipulator(original_data_container)
    modified_data_container = index_manipulator.get_timestamp_indexed_container()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # TODO Identify where the unclosed session originates from
