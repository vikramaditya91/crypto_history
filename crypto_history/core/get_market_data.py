from __future__ import annotations
import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from binance.client import Client
from typing import Union, Optional
from functools import lru_cache
from collections import namedtuple
from binance import enums
from dataclasses import make_dataclass
from .tickers import BinanceTickerPool, TickerPool
from .request import AbstractMarketRequester, BinanceRequester, CoinMarketCapRequester
from ..utilities.general_utilities import AbstractFactory, register_factory
logger = logging.getLogger(__name__)


class StockMarketFactory(AbstractFactory):
    """
    Abstract factory to generate factories per exchange
    """
    @abstractmethod
    def create_market_requester(self) -> AbstractMarketRequester:
        """
        Create the low-level market requester instance per exchange

        Returns: AbstractMarketRequester, instance of a market requester

        """
        pass

    @abstractmethod
    def create_market_operations(self, market_requester) -> AbstractMarketOperations:
        """
        Create the instance of the class to channel the right requests to the low level
        market requester.

        Args:
            market_requester: instance of AbstractMarketRequester

        Returns: instance of AbstractMarketOperations

        """
        pass

    @abstractmethod
    def create_data_homogenizer(self, market_operations) -> AbstractMarketHomogenizer:
        """
        Create an instance of a market homogenizer which is the interface to the market from the
        outside. It should ensure that the market data from various markets/exchanges
        which might have different low-level APIs is available in a uniform format.

        Args:
            market_operations: instance of AbstractMarketOperations

        Returns: AbstractMarketHomogenizer, instance of a market homogenizer

        """
        pass


@register_factory("market")
class ConcreteBinanceFactory:
    """Binance's factory"""

    def create_market_requester(self) -> BinanceRequester:
        """
        Creates the instance for the Binance Requester

        Returns: Instance of BinanceRequester

        """
        return BinanceRequester()

    def create_market_operations(self, market_requester) -> BinanceMarketOperations:
        return BinanceMarketOperations(market_requester)

    def create_data_homogenizer(self, market_operations) -> BinanceHomogenizer:
        return BinanceHomogenizer(market_operations)


@register_factory("market")
class ConcreteCoinMarketCapFactory(StockMarketFactory):
    def create_market_requester(self) -> CoinMarketCapRequester:
        return CoinMarketCapRequester()

    def create_market_operations(self, market_requester) -> CoinMarketCapMarketOperations:
        return CoinMarketCapMarketOperations(market_requester)

    def create_data_homogenizer(self, market_operations) -> CoinMarketCapHomogenizer:
        return CoinMarketCapHomogenizer(market_operations)


class AbstractMarketOperations(ABC):
    """Abstract MArket Operations"""
    def __init__(self, market_requester):
        """Init of market operations"""

        self.market_requester = market_requester

    @abstractmethod
    async def get_all_raw_tickers(self, *args, **kwargs):
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
                                     ticker: Union[str],
                                     interval: str,
                                     start_str: Union[str, datetime],
                                     end_str: Optional[Union[str, datetime]] = None,
                                     limit: Optional[int] = 500):
        if isinstance(start_str, datetime):
            start_str = str(start_str)
        end_str = end_str or datetime.now()
        if isinstance(end_str, datetime):
            end_str = str(end_str)
        binance_interval = self.match_binance_enum(interval)
        if limit>1000: #TODO Calculate correctly
            logger.warning("Limit exceeded. History is going to be truncated")
        return await self.market_requester.request("get_historical_klines",
                                                   ticker,
                                                   binance_interval,
                                                   start_str,
                                                   end_str,
                                                   limit)

    async def get_all_raw_tickers(self):
        return await self.market_requester.request("get_all_tickers")

    async def get_symbol_info(self, symbol):
        return await self.market_requester.request("get_symbol_info",
                                                   symbol)


class CoinMarketCapMarketOperations(AbstractMarketOperations):
    async def get_all_tickers(self, *args, **kwargs):
        raise NotImplementedError

    async def get_history_for_ticker(self, *args, **kwargs):
        raise NotImplementedError

    async def get_all_raw_tickers(self, *args, **kwargs):
        raise NotImplementedError


class AbstractMarketHomogenizer(ABC):
    History = None
    # TODO Make it an abstractmethod

    def __init__(self, market_operations):
        self.market_operator = market_operations

    @abstractmethod
    async def get_all_coins_ticker_objects(self) -> TickerPool:
        pass

    async def get_all_coins(self):
        pass

    @abstractmethod
    def get_ticker_instance(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_history_for_ticker(self, *args, **kwargs):
        pass


class BinanceHomogenizer(AbstractMarketHomogenizer):
    History = namedtuple("History", ["open_ts", "open", "high", "low", "close", "volume",
                                     "close_ts", "quote_asset_value", "number_of_trades",
                                     "taker_buy_base_asset_value", "take_buy_quote_asset_value",
                                     "ignored"])

    @lru_cache(maxsize=1)
    async def get_all_coins_ticker_objects(self) -> TickerPool:
        all_raw_tickers = await self.market_operator.get_all_raw_tickers()
        gathered_operations = []
        all_raw_tickers = all_raw_tickers[10:20]
        for raw_ticker in all_raw_tickers:
            gathered_operations.append(self.get_ticker_instance(raw_ticker['symbol']))
        all_tickers = await asyncio.gather(*gathered_operations, return_exceptions=False)
        return BinanceTickerPool(all_tickers)

    @staticmethod
    def get_ticker_dataclass(symbol_info_dict):
        return make_dataclass("Ticker", fields={k: type(v) for k, v in symbol_info_dict.items()},
                              eq=True, frozen=True)

    async def get_ticker_instance(self, ticker_name: str):
        symbol_info_dict = await self.market_operator.get_symbol_info(ticker_name)
        data_class_instance = self.get_ticker_dataclass(symbol_info_dict)
        return data_class_instance(**symbol_info_dict)

    async def get_base_reference_assets(self):
        ticker_pool = await self.get_all_coins_ticker_objects()
        base_assets = ticker_pool.obtain_unique_items("baseAsset")
        reference_assets = ticker_pool.obtain_unique_items("quoteAsset")
        return base_assets, reference_assets

    async def get_history_for_ticker(self, *args, **kwargs):
        raw_history = await self.market_operator.get_history_for_ticker(*args, **kwargs)
        return map(lambda x: self.History(*x), raw_history)


class CoinMarketCapHomogenizer(AbstractMarketHomogenizer):
    def get_ticker_instance(self, *args, **kwargs):
        pass

    def get_all_coins(self):
        pass

    def get_all_coins_ticker_objects(self) -> TickerPool:
        pass

    def get_history_for_ticker(self, *args, **kwargs):
        pass

