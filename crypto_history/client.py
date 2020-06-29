from .utilities import stock_market_abstract_factory
from datetime import datetime
from .utilities.general_utilities import init_logger
import logging


async def client_code() -> None:
    init_logger(level=logging.DEBUG)
    exchange_factory = stock_market_abstract_factory.ConcreteBinanceFactory()

    market_requester = exchange_factory.create_market_requester()

    market_operations = exchange_factory.create_market_operations(market_requester)
    all_tickers = await market_operations.get_all_tickers()
    a = await market_operations.get_history_for_ticker(all_tickers[0],
                                                       "5m",
                                                       start_str=datetime(2020, 4, 5))
    await market_requester.client.session.close()
    # pprint(a)

