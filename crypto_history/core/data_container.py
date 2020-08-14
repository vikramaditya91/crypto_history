from __future__ import annotations
import logging
import pandas as pd
import xarray as xr
from dataclasses import dataclass, fields
from typing import Union, List
from datetime import datetime
from abc import ABC, abstractmethod
from .tickers import TickerPool
from .get_market_data import StockMarketFactory
from ..utilities.general_utilities import register_factory
from ..utilities import general_utilities, exceptions

logger = logging.getLogger(__name__)


class DataFactory(general_utilities.AbstractFactory):
    """
    Generates the factories for the data aggregator
    """
    @abstractmethod
    def create_coin_history_obtainer(self) -> AbstractCoinHistoryObtainer:
        """
        Factory for creating the coin history obtainer which is the low-level interface to the market classes
        """
        pass

    @abstractmethod
    def create_data_container_operations(self, *args, **kwargs) -> AbstractDataContainerOperations:
        """
        Factory for creating the data container operations which is the interface
         for the user to access the stored data
        """
        pass

    @abstractmethod
    def create_data_container_dimensions_manager(self, *args, **kwargs) -> AbstractDimensionsManager:
        """
        Factory for managing the coordinates and the dimensions of the data-container
        """
        pass


@register_factory("data")
class ConcreteXArrayFactory(DataFactory):
    """
    Factory for generating the x-array related factories.
    XArray is the preferred way of storing multi-dimensional arrays
    """
    def create_coin_history_obtainer(self, *args, **kwargs) -> XArrayCoinHistoryObtainer:
        """
        Initialize the xarray coin history obtainer and return it

        Args:
            *args: see the class :class:`.AbstractCoinHistoryObtainer.__init__` for the arguments
            **kwargs: see the class :class:`.AbstractCoinHistoryObtainer.__init__` for the arguments

        Returns:
             XArrayCoinHistoryObtainer: initialized instance of the history obtainer suited for xarray

        """
        coin_history = XArrayCoinHistoryObtainer(*args, **kwargs)
        return coin_history

    def create_data_container_dimensions_manager(self, *args, **kwargs) -> XArrayDimensionsManager:
        return XArrayDimensionsManager(*args, **kwargs)

    def create_data_container_operations(self, history_obtainer, dimensions_manager) -> XArrayDataContainerOperations:
        """
        Creates the data container of the XArray that is meant to be accessible to the user

        Args:
            history_obtainer (XArrayCoinHistoryObtainer): history obtainer instance
            dimensions_manager (XArrayDimensionsManager): Manager of the dimensions/coordinates of the XArray

        Returns:
            XArrayDataContainerOperations: initialized instance of the DataContainer

        """
        return XArrayDataContainerOperations(history_obtainer, dimensions_manager)


@register_factory("data")
class ConcreteSomeOtherFactory(DataFactory):
    """Example of another factory for generating factories of data related class instances"""
    def create_coin_history_obtainer(self, *args, **kwargs):
        return SomeOtherCoinHistoryObtainer(*args, **kwargs)

    def create_data_container_operations(self, *args, **kwargs):
        return SomeOtherDataContainerOperations(*args, **kwargs)

    def create_data_container_dimensions_manager(self, *args, **kwargs):
        raise NotImplementedError


class AbstractCoinHistoryObtainer(ABC):
    """Abstract class to serve as the parent to generating histories obtianer classes"""
    def __init__(self,
                 exchange_factory: StockMarketFactory,
                 interval: str,
                 start_str: Union[str, datetime],
                 end_str: Union[str, datetime],
                 limit: int):
        """
        Generates the coin history obtainer

        Args:
            exchange_factory(StockMarketFactory): instance of the exchange_factory which is responsible for \
            setting the market_homogenizer
            interval(str): Length of the history of the klines per item
            start_str(str|datetime): duration from which history is necessary
            end_str(str|datetime): duration upto which history is necessary
            limit(int): number of klines required maximum. Note that it is limted by 1000 by binance
        """
        self.market_requester = exchange_factory.create_market_requester()
        self.market_operations = exchange_factory.create_market_operations(self.market_requester)
        self.market_harmonizer = exchange_factory.create_data_homogenizer(self.market_operations)
        self.data_container = None
        self.interval = interval
        self.start_str = start_str
        self.end_str = end_str
        self.limit = limit
        self.ticker_pool = None
        self.example_raw_history = None

    async def initialize_example(self):
        """
        Initializes an example of the raw history and stores it example_raw_history

        Returns:
             list: instance of an example of a raw history of ETHvsBTC

        """
        self.example_raw_history = self.example_raw_history or list(await self._get_raw_history_for_ticker("ETHBTC"))
        return self.example_raw_history

    async def _get_raw_history_for_ticker(self, ticker_symbol: str) -> map:
        """
        Gets the raw history for the ticker by using the interface to the market provided by the market homogenizer

        Args:
            ticker_symbol(str): Symbol of the ticker whose history is desired

        Returns:
             map: raw mapped history of the ticker mapped to the HistoryFields instance

        """
        return await self.market_harmonizer.get_history_for_ticker(ticker=ticker_symbol,
                                                                   interval=self.interval,
                                                                   start_str=self.start_str,
                                                                   end_str=self.end_str,
                                                                   limit=self.limit)

    async def get_ticker_pool(self) -> TickerPool:
        """
        Obtains the ticker pool by accessing the market homogenizer

        Returns:
             TickerPool: The TickerPool filled with all the ticker available to the market/exchange

        """
        if self.ticker_pool is None:
            self.ticker_pool = await self.market_harmonizer.get_all_coins_ticker_objects()
        return self.ticker_pool


class XArrayCoinHistoryObtainer(AbstractCoinHistoryObtainer):
    """The x-arrays's coin-history obtainer class"""
    async def get_historical_data_all_coins(self):
        """
        Get the history of all the tickers in the market/exchange

        Returns:
             map: map of all the histories in the exchange/market
             mapped from history to the HistoryFields

        """
        tasks_to_pursue = {}
        for ticker in self.ticker_pool:
            tasks_to_pursue[ticker.baseAsset, ticker.quoteAsset] = \
                self._get_raw_history_for_ticker(ticker.symbol)
        return await general_utilities.gather_dict(tasks_to_pursue)


class SomeOtherCoinHistoryObtainer(AbstractCoinHistoryObtainer):
    """An example of how another data-containers coin-history obtainer's factory would look like"""
    def get_all_inclusive_container(self, *args, **kwargs):
        pass


class AbstractDimensionsManager(ABC):
    @dataclass
    class Dimensions:
        """Class for keeping track of the dimensions of the XArray"""
        base_asset: list
        reference_asset: list
        field: list
        index_number: list

    """Class responsible for managing the dimensions/coordinates of teh data container"""
    def __init__(self, coin_history_obtainer):
        """Initializes the dimensions manager with the coin_history_obtainer"""
        self.coin_history_obtainer = coin_history_obtainer


class XArrayDimensionsManager(AbstractDimensionsManager):

    async def get_depth_of_indices(self):
        """
        Obtains the depth of the indices to identify how many time-stamp values
         each history has available

        Returns:
             list: The list of indices from 0..n-1
                     where n corresponds to the number of time-stamps in the history

        """
        example_history = await self.coin_history_obtainer.initialize_example()
        indices = list(range(len(list(example_history))))
        return indices

    async def get_mapped_coords(self):
        """Returns the mapped coordinates on the dimensions
        Returns:
            namedtuple: NamedTuple of the coordinates mapped onto dimensions
        """
        coordinates = await self.get_coords_for_data_array()
        return self.Dimensions(*coordinates)

    async def get_coords_for_data_array(self) -> List:
        """
        Gets the coordinates needed for the x-array.
        The coordinates for the data-array are arranged:
        coord1 => all base assets
        coord2 => all reference assets
        coord3 => all fields for each ticker (time stamp, OHLCV values, etc)
        coord1 => the depths of indices (number of time stamps) (0,1,2,..n-1) where n corresponds to each time stamp

        Returns:
             list: list of coordinates for the xarray

        """
        base_assets = await self.get_set_of_ticker_attributes("baseAsset")
        reference_assets = await self.get_set_of_ticker_attributes("quoteAsset")
        types_of_data_per_ts = self.coin_history_obtainer.market_harmonizer.get_all_history_fields()
        # TODO Use inheritance to avoid directly accessing private member
        return [list(base_assets),
                list(reference_assets),
                list(types_of_data_per_ts),
                await self.get_depth_of_indices()]

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
        for ticker in await self.coin_history_obtainer.get_ticker_pool():
            values.add(getattr(ticker, attribute))
        return values


class AbstractDataContainerOperations(ABC):
    """Abstract Base Class for generating the data operations"""
    def __init__(self, history_obtainer, dimensions_manager):
        """
        Initialize the data container operations

        Args:
            history_obtainer: instance of the history obtainer
            dimensions_manager: instance of the responsible class instance for managing dimensions/coordinates
        """
        self.history_obtainer = history_obtainer
        self.dimensions_manager = dimensions_manager
        self.data_container = None

    @abstractmethod
    async def get_all_inclusive_container(self, *args, **kwargs):
        """
        Obtain the filled history of the container
        """
        pass

    @abstractmethod
    def set_coords_dimensions_in_container(self):
        """
        Initialize the container (eg. with null-values) to avoid having
        to reshape the memory which can be processor intensive
        """
        pass


class XArrayDataContainerOperations(AbstractDataContainerOperations):
    async def set_coords_dimensions_in_container(self, dataclass_dimensions_coordinates=None):
        """
        Initialize the xarray container with None values but with the coordinates
        as it helps in the memory allocation
        Args:
            dataclass_dimensions_coordinates: dataclass of coordinates/dimensions as the framework for generating \
            the XArray
        """
        if dataclass_dimensions_coordinates is None:
            dataclass_dimensions_coordinates = await self.dimensions_manager.get_mapped_coords()
        dimensions = [dimension.name for dimension in fields(dataclass_dimensions_coordinates)]
        coordinates = [getattr(dataclass_dimensions_coordinates, item) for item in dimensions]
        self.data_container = xr.DataArray(None, coords=coordinates, dims=dimensions)

    async def get_populated_container(self, coord_dimension_dataclass):
        """
                Populates the container and returns it to the use

                Args:
                    coord_dimension_dataclass: coordinates of the XArray
                Returns:
                     xr.DataArray: the filled container of information

                """
        # TODO Avoid populating it every time it is called
        await self.set_coords_dimensions_in_container(coord_dimension_dataclass)
        await self.populate_container()
        return self.data_container

    async def get_all_inclusive_container(self):
        """
        Obtains the complete all-encompassing container
        Returns:
            xr.DataArray: the completely filled container

        """
        return await self.get_populated_container(coord_dimension_dataclass=None)

    async def populate_container(self):
        """
        Populates the xarray.DataArray with the historical data of all the coins
        """
        historical_data = await self.history_obtainer.get_historical_data_all_coins()
        for (base_asset, reference_asset), ticker_history in historical_data.items():
            try:
                history_df = await self.get_compatible_df(ticker_history)
            except exceptions.EmptyDataFrameException:
                continue
            self.data_container.loc[base_asset, reference_asset] = history_df.transpose()
            logger.debug(f"History set in x_array for ticker {base_asset}{reference_asset}")

    @staticmethod
    def calculate_rows_to_add(df, list_of_standard_history) -> int:
        """
        Calculates the additional number of rows that might have to be added to get the df in the same shape

        Args:
            df(pd.DataFrame): pandas dataframe which is obtained for the coin's history
            list_of_standard_history: expected standard history which has the complete history

        Returns:
             int: number of rows that have to be added to the df

        """
        df_rows, _ = df.shape
        expected_rows = len(list_of_standard_history)
        return expected_rows - df_rows

    async def get_compatible_df(self, ticker_history):
        """
        Makes the ticker history compatible to the standard pd.DataFrame by extending the
        shape to add null values

        Args:
            ticker_history: history of the current ticker

        Returns:
             pd.DataFrame: compatible history of the df adjusted for same rows as expected

        """
        # TODO Assuming that the df is only not filled in the bottom
        await self.history_obtainer.initialize_example()
        history_df = pd.DataFrame(ticker_history)
        if history_df.empty:
            raise exceptions.EmptyDataFrameException
        rows_to_add = self.calculate_rows_to_add(history_df, self.history_obtainer.example_raw_history)
        if rows_to_add > 0:
            history_df = self.add_extra_rows_to_bottom(history_df, rows_to_add)
        return history_df

    @staticmethod
    def add_extra_rows_to_bottom(df: pd.DataFrame, empty_rows_to_add: int):
        """
        Adds extra rows to the pd.DataFrame

        Args:
            df(pd.DataFrame): to which extra rows are to be added
            empty_rows_to_add(int): Number of extra rows that have to be added

        Returns:
             pd.DataFrame: Re-indexed pd.DataFrame which has the compatible number of rows

        """
        new_indices_to_add = list(range(df.index[-1] + 1, df.index[-1] + 1 + empty_rows_to_add))
        return df.reindex(df.index.to_list() + new_indices_to_add)


class SomeOtherDataContainerOperations(AbstractDataContainerOperations):
    """Example of how another data container operations's factory would look like"""
    def set_coords_dimensions_in_container(self):
        pass

    def get_all_inclusive_container(self, *args, **kwargs):
        pass
