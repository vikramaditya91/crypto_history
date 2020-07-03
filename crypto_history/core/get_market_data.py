from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from binance.client import Client
from .tickers import Ticker
from typing import Union, Optional
from binance import enums, client
from ..utilities import crypto_enum

logger = logging.getLogger(__package__)


class StockMarketFactory(ABC):
    @abstractmethod
    def create_market_requester(self) -> AbstractMarketRequester:
        pass

    @abstractmethod
    def create_market_operations(self) -> AbstractMarketOperations:
        pass

    @abstractmethod
    def create_data_homogenizer(self) -> AbstractMarketHomogenizer:
        pass


class ConcreteBinanceFactory(StockMarketFactory):
    def create_market_requester(self) -> BinanceRequester:
        return BinanceRequester()

    def create_market_operations(self, *args, **kwargs) -> BinanceMarketOperations:
        return BinanceMarketOperations(*args, **kwargs)

    def create_data_homogenizer(self, *args, **kwargs) -> BinanceHomogenizer:
        return BinanceHomogenizer(*args, **kwargs)


class ConcreteCoinMarketCapFactory(StockMarketFactory):
    def create_market_requester(self) -> CoinMarketCapRequester:
        return CoinMarketCapRequester()

    def create_market_operations(self, *args, **kwargs) -> CoinMarketCapMarketOperations:
        return CoinMarketCapMarketOperations(*args, **kwargs)

    def create_data_homogenizer(self, *args, **kwargs) -> CoinMarketCapHomogenizer:
        return CoinMarketCapHomogenizer(*args, **kwargs)


class AbstractMarketRequester(ABC):
    @abstractmethod
    def request(self, *args, **kwargs) -> str:
        pass


class BinanceRequester(AbstractMarketRequester):
    def __init__(self):
        self.client = client.AsyncClient(api_key="", api_secret="")

    async def request(self, binance_func: str, *args, **kwargs):

        logger.debug(f"Obtaining historical klines from Binance for:"
                     f"binance-function: {binance_func}\n"
                     f"args: {args}\t"
                     f"kwargs: {''.join(f'{key}: {value}' for key, value in kwargs)}")
        return await getattr(self.client, binance_func)(*args, **kwargs)


class CoinMarketCapRequester(AbstractMarketRequester):
    def __init__(self):
        raise NotImplementedError

    async def request(self, *args, **kwargs):
        raise NotImplementedError


class AbstractMarketOperations(ABC):
    def __init__(self, market_requester):
        self.market_requester = market_requester

    @abstractmethod
    async def get_all_tickers(self, *args, **kwargs):
        pass

    @abstractmethod
    async def get_history_for_ticker(self, *args, **kwargs):
        pass


class BinanceMarketOperations(AbstractMarketOperations):
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
        normalized_tickers = []
        for ticker in binance_tickers:
            normalized_tickers.append(Ticker.create_ticker_from_binance_item(ticker))
        # normalized_tickers = list(map(Ticker.create_ticker_from_binance_item, binance_tickers))
        return normalized_tickers


class CoinMarketCapMarketOperations(AbstractMarketOperations):
    async def get_all_tickers(self, *args, **kwargs):
        raise NotImplementedError

    async def get_history_for_ticker(self, *args, **kwargs):
        raise NotImplementedError


class AbstractMarketHomogenizer(ABC):
    def __init__(self, market_operations):
        self.market_requestor = market_operations

    async def get_all_coins(self) -> set:
        all_tickers = await self.market_requestor.get_all_tickers()
        coin_strings = set(Ticker.get_coin_str_from_ticker(ticker) for ticker in all_tickers
                           if ticker.ticker_name != crypto_enum.UNKNOWN_REFERENCE_COIN)
        return coin_strings


class BinanceHomogenizer(AbstractMarketHomogenizer):
    pass


class CoinMarketCapHomogenizer(AbstractMarketHomogenizer):
    pass


