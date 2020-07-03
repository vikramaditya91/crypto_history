import crypto_history.core.get_market_data
from datetime import datetime
from pprint import pprint
from .utilities.general_utilities import init_logger
import logging


async def client_code() -> None:
    init_logger(level=logging.DEBUG)
    exchange_factory = crypto_history.core.get_market_data.ConcreteBinanceFactory()

    market_requester = exchange_factory.create_market_requester()

    market_operations = exchange_factory.create_market_operations(market_requester)
    market_harmonizer = exchange_factory.create_data_homogenizer(market_operations)
    all_tickers = await market_harmonizer.get_all_coins()
    await market_requester.client.session.close()
    pprint(all_tickers)

