from datetime import datetime
from pprint import pprint
from .utilities.general_utilities import init_logger
import logging


async def client_code() -> None:
    init_logger(level=logging.DEBUG)
    market_factory = ConcreteBinanceFactory()
    data_container_factory = ConcreteXArrayFactory()
    coin_history_obtainer = await data_container_factory.create_coin_history_obtainer(market_factory,
                                                                                      interval="1d",
                                                                                      start_str="1 January 2020",
                                                                                      end_str="4 June 2020",
                                                                                      limit=1000
                                                                                      )
    data_operations = await data_container_factory.create_data_container_operations(coin_history_obtainer)
    data_container = await data_operations.get_filled_container()


    pprint(data_container)

