from __future__ import annotations
import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from binance.client import AsyncClient
from typing import Union, Optional, List, Dict, Generator, Set
from functools import lru_cache
from collections import namedtuple
from binance import enums
from dataclasses import make_dataclass
from .tickers import BinanceTickerPool, TickerPool
from .request import AbstractMarketRequester, BinanceRequester, SomeOtherExchangeRequester
from ..utilities.general_utilities import AbstractFactory, register_factory, get_dataclass_from_dict
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
    def create_market_operations() -> AbstractMarketOperations:
        """
        Create the instance of the class to channel the right requests to the low level
        market requester.

        """
        pass

    @staticmethod
    @abstractmethod
    def create_data_homogenizer() -> AbstractMarketHomogenizer:
        """
        Create an instance of a market homogenizer which is the interface to the market from the
        outside. It should ensure that the market data from various markets/exchanges
        which might have different low-level APIs is available in a uniform format.

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

    def create_market_operations(self) -> BinanceMarketOperations:
        """
        Creates the instance of the Binance Market Operator

        Returns:
             BinanceMarketOperations: Instance of BinanceMarketOperator

        """
        market_requester = self.create_market_requester()
        return BinanceMarketOperations(market_requester)

    def create_data_homogenizer(self) -> BinanceHomogenizer:
        """
        Creates the instance of the Binance Market Homogenizer

        Returns:
             BinanceHomogenizer: Instance of BinanceHomogenizer

        """
        market_operator = self.create_market_operations()
        return BinanceHomogenizer(market_operator)


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

    @abstractmethod
    async def get_exchange_info(self, *args, **kwargs):
        """Gets general properties of the exchange"""
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
             See details in :class:`.BinanceHomogenizer.OHLCVFields`

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

    async def get_exchange_info(self):
        """
        Obtains the complete information available at the exchange
        Returns:

        """
        return await self.market_requester.request("get_exchange_info")

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
    OHLCVFields = None
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

    def get_all_ohlcv_fields(self):
        """Gets all the fields from the historical named tuple"""
        return self.OHLCVFields._fields

    @abstractmethod
    def get_all_base_assets(self):
        pass

    @abstractmethod
    def get_all_reference_assets(self):
        pass

    @abstractmethod
    def get_all_raw_tickers(self):
        pass


class BinanceHomogenizer(AbstractMarketHomogenizer):
    OHLCVFields = namedtuple("OHLCVFields", ["open_ts", "open", "high", "low", "close", "volume",
                                                 "close_ts", "quote_asset_value", "number_of_trades",
                                                 "taker_buy_base_asset_value", "take_buy_quote_asset_value",
                                                 "ignored"])
    """Fields for the named tuple of the OHLCV returned by Binance get_klines_history
            See Also: https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#klinecandlestick-data
            See :meth:`binance.AsyncClient.get_historical_klines` in :py:mod:`python-binance`
    """

    async def get_exchange_assets(self, type_of_asset: str) -> Generator:
        """
        Obtains the type of asset from the exchange.
        Args:
            type_of_asset (str): string identifier that the binance API uses to identify the key in each symbol

        Returns:
            Generator: The generator of items that are available in the exchange

        """
        exchange_info = await self.get_exchange_info()
        symbols_on_exchange = exchange_info.symbols
        return (symbol[type_of_asset] for symbol in symbols_on_exchange)

    async def get_all_base_assets(self) -> List:
        """Obtains the list of all base assets """
        return list(set(await self.get_exchange_assets("baseAsset")))

    async def get_all_reference_assets(self) -> List:
        """Obtains the list of all reference assets"""
        return list(set(await self.get_exchange_assets("referenceAsset")))

    async def get_all_raw_tickers(self) -> List:
        """Obtains the list of all raw tickers available on binance"""
        a = await self.market_operator.get_all_raw_tickers()

        return await self.market_operator.get_all_raw_tickers()

    async def get_exchange_info(self):
        exchange_dict = await self.market_operator.get_exchange_info()
        exchange_dataclass = get_dataclass_from_dict("exchange_info", exchange_dict)
        return exchange_dataclass

    async def get_set_of_ticker_attributes(self, attribute: str):
        """
        Aggregates the set items of the required attribute
        across the whole history

        Args:
            attribute: The attribute/key which is the common term \
            in the history whose values are to be identified

        Returns:
             set: set of values whose attributes are common

        """

        values = set()
        for ticker in await self.get_all_coins_ticker_objects():
            values.add(getattr(ticker, attribute))
        return values

    @lru_cache(maxsize=1)
    async def get_all_coins_ticker_objects(self) -> TickerPool:
        """
        Obtains the exchange-independent TickerPool
        from all the tickers/symbols that are available on the exchange

        Returns:
             TickerPool: All tickers and their information stored in the TickerPool defined in
        # TODO

        """
        all_raw_tickers = await self.get_all_raw_tickers()
        gathered_operations = []
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
             map: map of the history of the ticker mapped to the OHLCVFields namedtuple

        """
        raw_history = await self.market_operator.get_raw_history_for_ticker(*args, **kwargs)
        return map(lambda x: self.OHLCVFields(*x), raw_history)


class SomeOtherExchangeHomogenizer(AbstractMarketHomogenizer):
    """Placeholder to show how another market homogenizer could be implemented"""
    def get_ticker_instance(self, *args, **kwargs):
        raise NotImplementedError

    def get_all_coins_ticker_objects(self) -> TickerPool:
        raise NotImplementedError

    def get_history_for_ticker(self, *args, **kwargs):
        raise NotImplementedError

