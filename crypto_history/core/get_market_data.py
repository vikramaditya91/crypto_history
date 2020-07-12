from __future__ import annotations
import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from binance.client import AsyncClient
from typing import Union, Optional, List, Dict
from functools import lru_cache
from collections import namedtuple
from binance import enums
from dataclasses import make_dataclass
from .tickers import BinanceTickerPool, TickerPool
from .request import AbstractMarketRequester, BinanceRequester, SomeOtherExchangeRequester
from ..utilities.general_utilities import AbstractFactory, register_factory
logger = logging.getLogger(__name__)


class StockMarketFactory(AbstractFactory):
    """
    Abstract factory to generate factories per exchange
    """

    @staticmethod
    @abstractmethod
    def create_market_requester() -> AbstractMarketRequester:
        """
        Create the low-level market requester instance per exchange
        """
        pass

    @staticmethod
    @abstractmethod
    def create_market_operations(market_requester) -> AbstractMarketOperations:
        """
        Create the instance of the class to channel the right requests to the low level
        market requester.

        Args:
            market_requester(AbstractMarketRequester): instance of AbstractMarketRequester
        """
        pass

    @staticmethod
    @abstractmethod
    def create_data_homogenizer(market_operations) -> AbstractMarketHomogenizer:
        """
        Create an instance of a market homogenizer which is the interface to the market from the
        outside. It should ensure that the market data from various markets/exchanges
        which might have different low-level APIs is available in a uniform format.

        Args:
            market_operations(AbstractMarketOperations): instance of AbstractMarketOperations
        """
        pass


@register_factory("market")
class ConcreteBinanceFactory:
    """Binance's factory for creating factories"""

    @staticmethod
    def create_market_requester() -> BinanceRequester:
        """
        Creates the instance for the Low Level Binance Requester

        Returns:
             BinanceRequester: Instance of BinanceRequester

        """
        return BinanceRequester()

    @staticmethod
    def create_market_operations(market_requester) -> BinanceMarketOperations:
        """
        Creates the instance of the Binance Market Operator
        
        Args:
            market_requester(BinanceRequester): BinanceRequester, Low level binance requester

        Returns:
             BinanceMarketOperations: Instance of BinanceMarketOperator

        """
        return BinanceMarketOperations(market_requester)

    @staticmethod
    def create_data_homogenizer(market_operations) -> BinanceHomogenizer:
        """
        Creates the instance of the Binance Market Homogenizer
        
        Args:
            market_operations(BinanceMarketOperations): instance of BinanceMarketOperations

        Returns:
             BinanceHomogenizer: Instance of BinanceHomogenizer

        """
        return BinanceHomogenizer(market_operations)


@register_factory("market")
class ConcreteSomeOtherExchangeFactory(StockMarketFactory):
    """Demo for how another exchange/market's factories would be implemented in this module"""

    @staticmethod
    def create_market_requester() -> SomeOtherExchangeRequester:
        """Creates the instance of another Market Requester"""
        raise NotImplementedError

    def create_market_operations(self, market_requester) -> SomeOtherExchangeMarketOperations:
        """Creates the instance of another Market's Operator"""
        raise NotImplementedError

    def create_data_homogenizer(self, market_operations) -> SomeOtherExchangeHomogenizer:
        """Creates the instance of the Market Homogenizer"""
        raise NotImplementedError


class AbstractMarketOperations(ABC):
    """Abstract Base Class to serve as the parent for all market operators.

    Seeing that the market requester may respond with different formats, it is not possible
    to know the signature of the method. The arguments and the return values are unknown
    by the AbstractMarketOperator

    """
    def __init__(self, market_requester):
        """
        The low level market requester which does the API calls

        Args:
            market_requester: instance of the market requester of the corresponding exchange
        """
        self.market_requester = market_requester

    @abstractmethod
    async def get_all_raw_tickers(self, *args, **kwargs):
        """
        Obtain all the tickers from the low-level market requester.
        Seeing that the market requester may respond with different formats, it is not possible
        to know the signature of the method
        """
        pass

    @abstractmethod
    async def get_raw_history_for_ticker(self, *args, **kwargs):
        """
        Obtain the raw history of a particular ticker.
        """
        pass

    @abstractmethod
    async def get_raw_symbol_info(self, *args, **kwargs):
        """
        Obtains the information on the particular ticker.
        It is not known what are the exact information that is going to be received
        because the information is coming from various exchanges
        """
        pass


class BinanceMarketOperations(AbstractMarketOperations):
    """Binance's market operator. Implements methods to get market knowledge"""

    @staticmethod
    def _match_binance_enum(string_to_match: str):
        """
        Convert the string to the python-binance API's enum
        :meth:`binance.enums` in :py:mod:`python-binance`
        https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.AsyncClient
        
        Args:
            string_to_match(str): string which should be matched to binance's enum

        Returns:
             binance.enum: enum object from python-binance

        """
        binance_matched_enum = list()
        for item in dir(enums):
            if getattr(enums, item) == string_to_match:
                binance_matched_enum.append(item)
        assert len(binance_matched_enum) == 1, f"Multiple Binance enums matched with {string_to_match}"
        return getattr(AsyncClient, binance_matched_enum[0])

    async def get_raw_history_for_ticker(self,
                                         ticker: Union[str],
                                         interval: str,
                                         start_str: Union[str, datetime],
                                         end_str: Optional[Union[str, datetime]] = None,
                                         limit: Optional[int] = 500) -> List:
        """
        Gets the kline history of the ticker from binance exchange

        Args:
            ticker (str): ticker whose history has to be pulled
            interval(str): interval of the history (eg. 1d, 3m, etc).\
            See :meth:`binance.enums` in :py:mod:`python-binance`
            start_str(str|datetime): Start date string in UTC format
            end_str(str|datetime): End date string in UTC format
            limit(int): list of klines


        Returns:
             list: List of snapshots of history.
             Each snapshot is a list of collection of OHLCV values.
             See details in :class:`.BinanceHomogenizer.HistoryFields`

        """
        # TODO This should probably be moved to the MarketHomogenizer
        if isinstance(start_str, datetime):
            start_str = str(start_str)
        end_str = end_str or datetime.now()
        if isinstance(end_str, datetime):
            end_str = str(end_str)
        binance_interval = self._match_binance_enum(interval)
        if limit>1000:
            # TODO Calculate correctly
            logger.warning("Limit exceeded. History is going to be truncated")
        return await self.market_requester.request("get_historical_klines",
                                                   ticker,
                                                   binance_interval,
                                                   start_str,
                                                   end_str,
                                                   limit)

    async def get_all_raw_tickers(self):
        """
        Gets all the tickers available on the binance exchange.
        

        Returns:
             list: All the raw tickers obtained from the python-binance (:py:mod:`python-binance`)\
             In binance, the format of each raw ticker is {"symbol": <>, "price": <>}

        """
        return await self.market_requester.request("get_all_tickers")

    async def get_raw_symbol_info(self, symbol: str):
        """
        Obtains the information for the symbol/ticker requested

        Args:
            symbol(str): symbol of the ticker whose information is desired

        Returns:
             dict: The raw information of the ticker desired with information where the keys are the
             baseAsset, precision, quoteAsset, etc.
             See :meth:`binance.AsyncClient.get_symbol_info` in :py:mod:`python-binance
             <https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.Client>`

        """
        return await self.market_requester.request("get_symbol_info",
                                                   symbol)


class SomeOtherExchangeMarketOperations(AbstractMarketOperations):
    """Place holder for another exchange that could be integrated in this module"""
    def get_raw_symbol_info(self, *args, **kwargs):
        raise NotImplementedError

    async def get_raw_history_for_ticker(self, *args, **kwargs):
        raise NotImplementedError

    async def get_all_raw_tickers(self, *args, **kwargs):
        raise NotImplementedError


class AbstractMarketHomogenizer(ABC):
    """Synthesizes the information obtained from various different market operator to a consistent format"""
    HistoryFields = None
    # TODO Make it an abstractmethod

    def __init__(self, market_operations):
        """
        Initializes the class with the instance of the corresponding market operator

        Args:
            market_operations (AbstractMarketOperations): Instance of the corresponding market operator
        """
        self.market_operator = market_operations

    @abstractmethod
    async def get_all_coins_ticker_objects(self) -> TickerPool:
        """
        Generates the standard/uniform TickerPool object which should be consistent no matter which
        exchange it is coming from

        Returns:
             TickerPool: TickerPool which contains all the different tickers and holds it in one

        """
        pass

    @abstractmethod
    def get_ticker_instance(self, *args, **kwargs):
        """
        Gets the standard ticker dataclass object.
        Note that the fields may be dependent on the exchange that it is coming from

        Args:
            *args: unknown, as it depends on the exchange
            **kwargs: unknown, as it depends on the exchange

        Returns:
             dataclass: dataclass instance of the symbol info

        """
        pass

    @abstractmethod
    def get_history_for_ticker(self, *args, **kwargs) -> map:
        """
        Gets the history of the ticker symbol

        Args:
            *args: unknown, as it depends on the exchange
            **kwargs: unknown, as it depends on the exchange

        Returns:
             map: map object of history of the ticker

        """
        pass


class BinanceHomogenizer(AbstractMarketHomogenizer):
    HistoryFields = namedtuple("HistoryFields", ["open_ts", "open", "high", "low", "close", "volume",
                                                 "close_ts", "quote_asset_value", "number_of_trades",
                                                 "taker_buy_base_asset_value", "take_buy_quote_asset_value",
                                                 "ignored"])
    """Fields for the named tuple of the OHLCV returned by Binance get_klines_history
            See :meth:`binance.AsyncClient.get_historical_klines` in :py:mod:`python-binance`
    """

    @lru_cache(maxsize=1)
    async def get_all_coins_ticker_objects(self) -> TickerPool:
        """
        Obtains the exchange-independent TickerPool
        from all the tickers/symbols that are available on the exchange

        Returns:
             TickerPool: All tickers and their information stored in the TickerPool defined in
        # TODO

        """
        all_raw_tickers = await self.market_operator.get_all_raw_tickers()
        gathered_operations = []
        all_raw_tickers = all_raw_tickers[10:20]
        for raw_ticker in all_raw_tickers:
            gathered_operations.append(self.get_ticker_instance(raw_ticker['symbol']))
        all_tickers = await asyncio.gather(*gathered_operations, return_exceptions=False)
        return BinanceTickerPool(all_tickers)

    @staticmethod
    def _get_ticker_dataclass(symbol_info_dict: Dict):
        """
        Generates a dataclass from the dictionary of the raw ticker provided

        Args:
            symbol_info_dict: dictionary of the raw ticker

        Returns:
             Ticker: A dataclass instance which holds the symbol information

        """
        return make_dataclass("Ticker", fields={k: type(v) for k, v in symbol_info_dict.items()},
                              eq=True, frozen=True)

    async def get_ticker_instance(self, ticker_name: str):
        """
        Obtains the TickerDataclass based on the string of the ticker name provided
        
        Args:
            ticker_name(str): ticker name whose standard Ticker dataclass object is desired

        Returns:
             dataclass: instance of the dataclass of the ticker

        """
        symbol_info_dict = await self.market_operator.get_raw_symbol_info(ticker_name)
        data_class_instance = self._get_ticker_dataclass(symbol_info_dict)
        return data_class_instance(**symbol_info_dict)

    async def get_history_for_ticker(self, *args, **kwargs) -> map:
        """
        Gets the history of the ticker for the desired duration,

        # TODO Probably yank the doc from MarketOperator to here

        Args:
            *args: See the :class:`.BinanceMarketOperator.get_raw_history_for_ticker` for arguments
            **kwargs: See the :class:`.BinanceMarketOperator.get_raw_history_for_ticker` for arguments

        Returns:
             map: map of the history of the ticker mapped to the HistoryFields namedtuple

        """
        raw_history = await self.market_operator.get_raw_history_for_ticker(*args, **kwargs)
        return map(lambda x: self.HistoryFields(*x), raw_history)


class SomeOtherExchangeHomogenizer(AbstractMarketHomogenizer):
    """Placeholder to show how another market homogenizer could be implemented"""
    def get_ticker_instance(self, *args, **kwargs):
        raise NotImplementedError

    def get_all_coins_ticker_objects(self) -> TickerPool:
        raise NotImplementedError

    def get_history_for_ticker(self, *args, **kwargs):
        raise NotImplementedError

