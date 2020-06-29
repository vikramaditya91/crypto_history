import logging
from abc import ABC, abstractmethod
from datetime import datetime
from binance.client import Client
from .tickers import Ticker
from typing import Union, Optional
from binance import enums

logger = logging.getLogger(__package__)


class AbstractMarketOperations(ABC):
    @abstractmethod
    async def get_all_tickers(self, *args, **kwargs):
        pass

    @abstractmethod
    async def get_history_for_ticker(self, *args, **kwargs):
        pass


class BinanceMarketOperations(AbstractMarketOperations):
    def __init__(self, market_requester):
        self.market_requester = market_requester

    @staticmethod
    def match_binance_enum(string_to_match: str):
        binance_matched_enum = list()
        for item in dir(enums):
            if getattr(enums, item) == string_to_match:
                binance_matched_enum.append(item)
        assert len(binance_matched_enum) == 1, f"Multiple Binance enums matched with {string_to_match}"
        return getattr(Client, binance_matched_enum[0])

    async def get_history_for_ticker(self,
                                     ticker: Union[str, Ticker],
                                     interval: str,
                                     start_str: Union[str, datetime],
                                     end_str: Optional[Union[str, datetime]] = None,
                                     limit: Optional[int] = 500):
        if isinstance(ticker, Ticker):
            ticker = ticker.ticker_name
        if isinstance(start_str, datetime):
            start_str = str(start_str)
        end_str = end_str or datetime.now()
        if isinstance(end_str, datetime):
            end_str = str(end_str)
        binance_interval = self.match_binance_enum(interval)
        return await self.market_requester.request("get_historical_klines",
                                                   ticker,
                                                   binance_interval,
                                                   start_str,
                                                   end_str,
                                                   limit)

    async def get_all_tickers(self, *args, **kwargs):
        binance_tickers = await self.market_requester.request("get_all_tickers")
        normalized_tickers = list(map(Ticker.create_ticker_from_binance_item, binance_tickers))
        return normalized_tickers

    async def get_all_coins(self):
        all_coins = await self.market_requester.request()


class CoinMarketCapMarketOperations(AbstractMarketOperations):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    async def get_all_tickers(self, *args, **kwargs):
        raise NotImplementedError

    async def get_history_for_ticker(self, *args, **kwargs):
        raise NotImplementedError

# class AbstractMarketHarmonizer(ABC):
#     @abstractmethod
#     def get_all_tickers(self, *args, **kwargs) -> None:
#         pass
#
#
# class BinanceHarmonizer(AbstractMarketHarmonizer):
#     def get_all_tickers(self, historical_data):
#         print(f"Historical data is being cleaned on A")
#
#
# class CoinMarketCapHarmonizer(AbstractMarketHarmonizer):
#     def get_all_tickers(self, historical_data):
#         print(f"Historical data is being cleaned on B")