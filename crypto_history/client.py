from datetime import datetime
from pprint import pprint
from .utilities.general_utilities import init_logger
import logging
from .core.data_container import ConcreteXArrayFactory, ConcreteBinanceFactory


async def client_code() -> None:
    init_logger(level=logging.DEBUG)
    market_factory = ConcreteBinanceFactory()
    data_container_factory = ConcreteXArrayFactory()

    coin_history_container = await data_container_factory.create_container_coin_history(market_factory,
                                                                                        interval="1d",
                                                                                        start_str="1 January 2020",
                                                                                        end_str="4 June 2020",
                                                                                        limit=1000)
    await coin_history_container.get_filled_container()

    # pprint(all_base_assets)
    # pprint(all_reference_assets)

