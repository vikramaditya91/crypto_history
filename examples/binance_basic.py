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
                                                                                start_str="1 January 2020",
                                                                                end_str="4 June 2020",
                                                                                limit=1000
                                                                                )
    container_dimensions_manager = data_container_factory.\
        create_data_container_dimensions_manager(coin_history_obtainer)
    coord_dimension_dataclass = await container_dimensions_manager.get_mapped_coords()
    print(f"Available coordinates of the XArray are {coord_dimension_dataclass}\n")
    desired_fields = ['open_ts', 'open', 'high', 'close', 'close_ts']
    reference_asset = ["BTC"]
    desired_dataclass = replace(coord_dimension_dataclass,
                                reference_asset=reference_asset,
                                field=desired_fields)

    data_operations = data_container_factory.create_data_container_operations(coin_history_obtainer,
                                                                              container_dimensions_manager)
    data_container = await data_operations.get_populated_container(coord_dimension_dataclass=desired_dataclass)
    pprint(data_container)
    # TODO Identify where the unclosed session originates from

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())