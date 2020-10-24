from __future__ import annotations
import logging
import asyncio
import math
import datetime
import contextlib
from dateutil import parser
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Generator
from functools import lru_cache
from collections import namedtuple
from binance import enums, client
from dataclasses import make_dataclass
from pydoc import locate
from crypto_history.stock_market.tickers import BinanceTickerPool, TickerPool
from crypto_history.stock_market.request import (
    AbstractMarketRequester,
    BinanceRequester,
    SomeOtherExchangeRequester,
)
from crypto_history.utilities.general_utilities import (
    AbstractFactory,
    register_factory,
    get_dataclass_from_dict,
)
from crypto_history.utilities.datetime_operations import \
    DateTimeOperations

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
        Create the instance of the class to channel the right requests\
         to the low level market requester.

        """
        pass

    @staticmethod
    @abstractmethod
    def create_data_homogenizer() -> AbstractMarketHomogenizer:
        """
        Create an instance of a market homogenizer which is the interface\
         to the market from the outside. It should ensure that the market\
          data from various markets/exchanges which might have different \
          low-level APIs is available in a uniform format.

        """
        pass

    @staticmethod
    @abstractmethod
    def create_ohlcv_field_types() -> AbstractOHLCVFieldTypes:
        pass

    @staticmethod
    @abstractmethod
    def create_time_interval_chunks() -> AbstractTimeIntervalChunks:
        pass


@register_factory(section="market", identifier="binance")
class ConcreteBinanceFactory(StockMarketFactory):
    """Binance's factory for creating factories"""

    @staticmethod
    def create_market_requester() -> BinanceRequester:
        """
        Creates the instance for the Low Level Binance Requester

        Returns:
             BinanceRequester: Instance of BinanceRequester

        """
        return BinanceRequester()

    @contextlib.asynccontextmanager
    async def create_market_operations(self) -> BinanceMarketOperations:
        """
        Creates the instance of the Binance Market Operator

        Returns:
             BinanceMarketOperations: Instance of BinanceMarketOperator

        """
        async with self.create_market_requester() as market_requester:
            yield BinanceMarketOperations(market_requester)

    @contextlib.asynccontextmanager
    async def create_data_homogenizer(self) -> BinanceHomogenizer:
        """
        Creates the instance of the Binance Market Homogenizer

        Returns:
             BinanceHomogenizer: Instance of BinanceHomogenizer

        """
        async with self.create_market_operations() as market_operator:
            type_checker = self.create_ohlcv_field_types()
            yield BinanceHomogenizer(market_operator, type_checker)

    @staticmethod
    def create_ohlcv_field_types() -> BinanceOHLCVFieldTypes:
        """
        Creates the instance of the Binance OHLCV Fields Types

        Returns:
            BinanceOHLCVFieldTypes: Instance of BinanceOHLCVFieldTypes

        """
        return BinanceOHLCVFieldTypes()

    @staticmethod
    def create_time_interval_chunks() -> BinanceTimeIntervalChunks:
        """
        Creates the instance of the Binance time interval chunks

        Returns:
            BinanceTimeIntervalChunks: Instance of BinanceTimeIntervalChunks

        """
        return BinanceTimeIntervalChunks()


@register_factory(section="market", identifier="some_other_exchange")
class ConcreteSomeOtherExchangeFactory(StockMarketFactory):
    """Demo for how another exchange/market's factories would be\
     implemented in this module"""

    @staticmethod
    def create_market_requester() -> SomeOtherExchangeRequester:
        """Creates the instance of another Market Requester"""
        raise NotImplementedError

    @staticmethod
    def create_market_operations() -> SomeOtherExchangeMarketOperations:
        """Creates the instance of another Market's Operator"""
        raise NotImplementedError

    @staticmethod
    def create_data_homogenizer() -> SomeOtherExchangeHomogenizer:
        """Creates the instance of the Market Homogenizer"""
        raise NotImplementedError

    @staticmethod
    def create_ohlcv_field_types() -> SomeOtherOHLCVFieldTypes:
        raise NotImplementedError

    @staticmethod
    def create_time_interval_chunks() -> SomeOtherTimeIntervalChunks:
        raise NotImplementedError


class AbstractMarketOperations(ABC):
    """Abstract Base Class to serve as the parent for all market operators.

    Seeing that the market requester may respond with different formats,\
     it is not possible to know the signature of the method. \
     The arguments and the return values are unknown by the \
     AbstractMarketOperator

    """

    def __init__(self, market_requester):
        """
        The low level market requester which does the API calls

        Args:
            market_requester: instance of the market requester of\
             the corresponding exchange
        """
        self.market_requester = market_requester

    @abstractmethod
    async def get_all_raw_tickers(self, *args, **kwargs):
        """
        Obtain all the tickers from the low-level market requester.
        Seeing that the market requester may respond with different\
         formats, it is not possible to know the signature of the method
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
        It is not known what are the exact information that is\
         going to be received because the information is coming \
         from various exchanges
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
        https://python-binance.readthedocs.io/en/latest/\
        binance.html#binance.client.AsyncClient

        Args:
            string_to_match(str): string which should be matched\
             to binance's enum

        Returns:
             binance.enum: enum object from python-binance

        """
        binance_matched_enum = list()
        for item in dir(enums):
            if getattr(enums, item) == string_to_match:
                binance_matched_enum.append(item)
        assert (
            len(binance_matched_enum) == 1
        ), f"Multiple Binance enums matched with {string_to_match}"
        return getattr(client.AsyncClient, binance_matched_enum[0])

    async def get_raw_history_for_ticker(
        self,
        ticker: Union[str],
        interval: str,
        start_time: int,
        end_time: int,
    ) -> List:
        """
        Gets the kline history of the ticker from binance exchange

        Args:
            ticker (str): ticker whose history has to be pulled
            interval(str): interval of the history (eg. 1d, 3m, etc).\
            See :meth:`binance.enums` in :py:mod:`python-binance`
            start_time(int): Start date string in exchange-format
            end_time(int): End date string in exchange-format

        Returns:
             list: List of snapshots of history.
             Each snapshot is a list of collection of OHLCV values.
             See details in :class:`.BinanceHomogenizer.OHLCVFields`

        """
        binance_interval = self._match_binance_enum(interval)
        return await self.market_requester.request(
            "get_historical_klines",
            ticker,
            binance_interval,
            start_time,
            end_time,
        )

    async def get_all_raw_tickers(self):
        """
        Gets all the tickers available on the binance exchange.

        Returns:
             list: All the raw tickers obtained from the python-binance\
              (:py:mod:`python-binance`)\
             In binance, the format of each raw ticker is \
             {"symbol": <>, "price": <>}

        """
        return await self.market_requester.request("get_all_tickers")

    async def get_raw_symbol_info(self, symbol: str):
        """
        Obtains the information for the symbol/ticker requested

        Args:
            symbol(str): symbol of the ticker whose information is desired

        Returns:
             dict: The raw information of the ticker desired with \
             information where the keys are the
             baseAsset, precision, quoteAsset, etc.
             See :meth:`binance.AsyncClient.get_symbol_info` in \
             :py:mod:`python-binance
             <https://python-binance.readthedocs.io/en/latest/\
             binance.html#binance.client.Client>`

        """
        return await self.market_requester.request("get_symbol_info", symbol)

    async def get_exchange_info(self):
        """
        Obtains the complete information available at the exchange
        Returns:

        """
        return await self.market_requester.request("get_exchange_info")


class SomeOtherExchangeMarketOperations(AbstractMarketOperations):
    """Place holder for another exchange that could be \
    integrated in this module"""

    def get_raw_symbol_info(self, *args, **kwargs):
        raise NotImplementedError

    async def get_raw_history_for_ticker(self, *args, **kwargs):
        raise NotImplementedError

    async def get_all_raw_tickers(self, *args, **kwargs):
        raise NotImplementedError

    def get_exchange_info(self, *args, **kwargs):
        raise NotImplementedError


class AbstractMarketHomogenizer(ABC):
    """Synthesizes the information obtained from various different market\
     operator to a consistent format"""

    OHLCVFields = None
    # TODO Make it an abstractmethod

    def __init__(self, market_operations, type_checker):
        """
        Initializes the class with the instance of the corresponding \
        market operator

        Args:
            market_operations (AbstractMarketOperations): Instance of \
            the corresponding market operator
        """
        self.market_operator = market_operations
        self.type_checker = type_checker

    @abstractmethod
    async def get_all_coins_ticker_objects(self) -> TickerPool:
        """
        Generates the standard/uniform TickerPool object which should \
        be consistent no matter which exchange it is coming from

        Returns:
             TickerPool: TickerPool which contains all the different \
             tickers and holds it in one

        """
        pass

    @abstractmethod
    def get_ticker_instance(self, *args, **kwargs):
        """
        Gets the standard ticker dataclass object.
        Note that the fields may be dependent on the exchange \
        that it is coming from

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

    @abstractmethod
    def get_all_base_assets(self):
        pass

    @abstractmethod
    def get_all_reference_assets(self):
        pass

    @abstractmethod
    def get_all_raw_tickers(self):
        pass

    def get_named_ohlcv_tuple(self):
        return self.type_checker.get_named_tuple()


class BinanceHomogenizer(AbstractMarketHomogenizer):
    async def get_exchange_assets(self, type_of_asset: str) -> Generator:
        """
        Obtains the type of asset from the exchange.
        Args:
            type_of_asset (str): string identifier that the binance\
             API uses to identify the key in each symbol

        Returns:
            Generator: The generator of items that are available in\
             the exchange

        """
        exchange_info = await self.get_exchange_info()
        symbols_on_exchange = exchange_info.symbols
        return (symbol[type_of_asset] for symbol in symbols_on_exchange)

    async def get_all_base_assets(self) -> List:
        """Obtains the list of all base assets """
        return list(set(await self.get_exchange_assets("baseAsset")))

    async def get_all_reference_assets(self) -> List:
        """Obtains the list of all reference assets"""
        return list(set(await self.get_exchange_assets("quoteAsset")))

    async def get_all_raw_tickers(self) -> List:
        """Obtains the list of all raw tickers available on binance"""
        return await self.market_operator.get_all_raw_tickers()

    async def get_exchange_info(self):
        exchange_dict = await self.market_operator.get_exchange_info()
        exchange_dataclass = get_dataclass_from_dict(
            "exchange_info", exchange_dict
        )
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
             TickerPool: All tickers and their information stored in the\
              TickerPool defined in
        # TODO

        """
        all_raw_tickers = await self.get_all_raw_tickers()
        gathered_operations = []
        for raw_ticker in all_raw_tickers:
            gathered_operations.append(
                self.get_ticker_instance(raw_ticker["symbol"])
            )
        all_tickers = await asyncio.gather(
            *gathered_operations, return_exceptions=False
        )
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
        return make_dataclass(
            "Ticker",
            fields={k: type(v) for k, v in symbol_info_dict.items()},
            eq=True,
            frozen=True,
        )

    async def get_ticker_instance(self, ticker_name: str):
        """
        Obtains the TickerDataclass based on the string of the \
        ticker name provided

        Args:
            ticker_name(str): ticker name whose standard Ticker \
            dataclass object is desired

        Returns:
             dataclass: instance of the dataclass of the ticker

        """
        symbol_info_dict = await self.market_operator.get_raw_symbol_info(
            ticker_name
        )
        data_class_instance = self._get_ticker_dataclass(symbol_info_dict)
        return data_class_instance(**symbol_info_dict)

    async def get_history_for_ticker(self, *args, **kwargs) -> map:
        """
        Gets the history of the ticker for the desired duration,

        # TODO Probably yank the doc from MarketOperator to here

        Args:
            *args: See the :class:`.BinanceMarketOperator.\
            get_raw_history_for_ticker` for arguments
            **kwargs: See the :class:`.BinanceMarketOperator.\
            get_raw_history_for_ticker` for arguments

        Returns:
             map: map of the history of the ticker mapped to \
             the OHLCVFields namedtuple

        """
        raw_history = await self.market_operator.get_raw_history_for_ticker(
            *args, **kwargs
        )
        named_tuple_instance = self.get_named_ohlcv_tuple()
        return map(lambda x: named_tuple_instance(*x), raw_history)


class SomeOtherExchangeHomogenizer(AbstractMarketHomogenizer):
    """Placeholder to show how another market\
     homogenizer could be implemented"""

    def get_ticker_instance(self, *args, **kwargs):
        raise NotImplementedError

    def get_all_coins_ticker_objects(self) -> TickerPool:
        raise NotImplementedError

    def get_history_for_ticker(self, *args, **kwargs):
        raise NotImplementedError

    def get_all_reference_assets(self):
        raise NotImplementedError

    def get_all_base_assets(self):
        raise NotImplementedError

    def get_all_raw_tickers(self):
        raise NotImplementedError


class AbstractOHLCVFieldTypes(ABC):
    class OHLCVFields:
        pass

    # Do not consider a dataclass as I could not
    # find a quick way to get data in a df from data
    def get_named_tuple(self):
        return namedtuple(
            self.OHLCVFields.__name__, self.OHLCVFields.__annotations__.keys()
        )

    def get_dict_name_type(self):
        return dict(
            map(
                lambda x: (x[0], locate(x[1])),
                self.OHLCVFields.__annotations__.items(),
            )
        )


class BinanceOHLCVFieldTypes(AbstractOHLCVFieldTypes):
    """Fields for the named tuple of the OHLCV returned by Binance \
    get_klines_history
            See Also: https://github.com/binance-exchange/\
            binance-official-api-docs/blob/master/\
            rest-api.md#klinecandlestick-data
            See :meth:`binance.AsyncClient.get_historical_klines` in \
            :py:mod:`python-binance`
    """

    class OHLCVFields:
        open_ts: int
        open: float
        high: float
        low: float
        close: float
        volume: float
        close_ts: int
        quote_asset_value: float
        number_of_trades: int
        taker_buy_base_asset_value: float
        take_buy_quote_asset_value: float
        ignored: int


class SomeOtherOHLCVFieldTypes(AbstractOHLCVFieldTypes):
    pass


class AbstractTimeIntervalChunks(ABC):
    """Abstract class to handle the chunking of dates as the \
    exchanges have limits on the length of the interval of history"""

    limit = math.inf
    url = ""

    def get_time_range_for_historical_calls(
        self, raw_time_range_dict: Dict
    ) -> List[tuple]:
        """
        Obtains the time ranges for making the historical calls.

        Args:
            raw_time_range_dict: Dictionary of the
            (start:str, end:str): (type_of_interval: str)

        Returns:
            List of the tuples of the exchange-specific format.
            ((start_time, end_time), type_of_interval)

        """
        final_time_range = []
        datetime_operations = DateTimeOperations()
        for (
            (start_time, end_time),
            type_of_interval,
        ) in raw_time_range_dict.items():
            if isinstance(start_time, str):
                start_time = self.sanitize_item_to_datetime_object(start_time)
            if isinstance(end_time, str):
                end_time = self.sanitize_item_to_datetime_object(end_time)
            sanitized_kline_width = datetime_operations.\
                map_string_to_timedelta(
                    type_of_interval
                )
            sub_chunks = self._get_chunks_from_start_end_complete(
                start_time, end_time, sanitized_kline_width
            )
            sanitized_sub_chunks = self.get_exchange_specific_sub_chunks(
                sub_chunks
            )
            [
                final_time_range.append((chunk, type_of_interval))
                for chunk in sanitized_sub_chunks
            ]
        logger.info(
            f"The time histories have been chunked into" f" {final_time_range}"
        )
        return final_time_range

    def get_exchange_specific_sub_chunks(
        self, sub_chunks: List[tuple]
    ) -> List[tuple]:
        """
        Gets the exchange specific sub-chunk from default sub-chunks
        Args:
            sub_chunks (list): default sub-chunks to be converted \
            to exchange specific

        Returns:
            List of exchange-specific formatted sub-chunks

        """
        list_of_sub_chunks = []
        for sub_chunk in sub_chunks:
            list_of_sub_chunks.append(
                tuple(
                    map(self.sanitize_datetime_to_exchange_specific, sub_chunk)
                )
            )
        return list_of_sub_chunks

    @staticmethod
    def sanitize_item_to_datetime_object(
        item_to_parse: str,
    ) -> datetime.datetime:
        """
        Converts/sanitizes the string to the datetime.datetime object
        Notes:
            Timezone is not handled. The local timezone is considered \
            by dateutil's parser
        Args:
            item_to_parse (str): the item that has to be converted

        Returns:
            datetime.datetime object from the string

        """
        if item_to_parse == "now":
            return datetime.datetime.now()
        return parser.parse(item_to_parse)

    @staticmethod
    def _calculate_klines_in_interval(
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        timedelta_kline: datetime.timedelta,
    ) -> int:
        """
        Calculates the expected klines from the start and end dates
        Args:
            start_datetime (datetime.datetime): start time
            end_datetime (datetime.datetime): end time
            timedelta_kline (datetime.timedelta): the interval of each kline

        Returns:
            The number of klines expected between these times

        """
        length_of_interval = end_datetime - start_datetime
        return math.ceil(length_of_interval / timedelta_kline)

    @staticmethod
    def _get_chunks_from_number_of_intervals(
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        number_of_intervals: int,
    ) -> List[tuple]:
        """
        Gets the chunks by splitting the times into the number of intervals
        Args:
            start_datetime (datetime.datetime): start of the overall time
            end_datetime (datetime.datetime): end of the overall time
            number_of_intervals (int): the number of intervals between the \
             times

        Returns:
            List[tuple] the tuple is set as (start_ts_of_sub_chunk, \
            end_ts_of_sub_chunk)

        """
        chunks_of_intervals = []
        average_timedelta = (
            end_datetime - start_datetime
        ) / number_of_intervals
        for i in range(number_of_intervals):
            chunks_of_intervals.append(
                (
                    start_datetime + average_timedelta * i,
                    start_datetime + average_timedelta * (i + 1),
                )
            )
        # There could possibly be a difference in the end-time of the
        # last time item due to round-off errors
        return chunks_of_intervals

    def _get_chunks_from_start_end_complete(
        self,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        interval_kline: datetime.timedelta,
    ) -> List[tuple]:
        """
        Gets the chunks based on the intervals of the timedelta
        Args:
            start_datetime (datetime.datetime): start of the overall time
            end_datetime (datetime.datetime): end of the overall time
            interval_kline (datetime.timedelta): the timedelta of each kline

        Returns:
            List[tuple] the tuple is set as (start_ts_of_sub_chunk, \
            end_ts_of_sub_chunk)

        """
        number_of_klines = self._calculate_klines_in_interval(
            start_datetime, end_datetime, interval_kline
        )
        number_of_intervals_necessary = math.ceil(
            number_of_klines / self.limit
        )

        return self._get_chunks_from_number_of_intervals(
            start_datetime, end_datetime, number_of_intervals_necessary
        )

    @staticmethod
    @abstractmethod
    def sanitize_datetime_to_exchange_specific(
        datetime_obj: datetime.datetime,
    ):
        """Converts the datetime object to exchange specific format"""
        pass


class BinanceTimeIntervalChunks(AbstractTimeIntervalChunks):
    """Binance specific information for the interval generation"""

    url = "https://github.com/binance-exchange/binance-\
    official-api-docs/blob/master/rest-api.md#enum-definitions"
    limit = 1000

    @staticmethod
    def sanitize_datetime_to_exchange_specific(
        datetime_obj: datetime.datetime,
    ) -> int:
        """
        Converts the datetime object to binance specific format for \
        making the requests
        Args:
            datetime_obj: datetime.datetime object which needs to be converted

        Returns:
            binance specific format

        """
        return int(datetime_obj.timestamp()) * 1000


class SomeOtherTimeIntervalChunks(AbstractTimeIntervalChunks):
    @staticmethod
    def sanitize_datetime_to_exchange_specific(datetime_obj):
        pass
