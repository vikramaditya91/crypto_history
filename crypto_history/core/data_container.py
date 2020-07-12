from __future__ import annotations
import logging
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
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
        pass

    @abstractmethod
    def create_data_container_operations(self) -> AbstractDataContainerOperations:
        pass


@register_factory("data")
class ConcreteXArrayFactory(DataFactory):
    async def create_coin_history_obtainer(self, *args, **kwargs):
        coin_history = XArrayCoinHistoryObtainer(*args, **kwargs)
        coin_history.ticker_pool = await coin_history.initialize_ticker_pool()
        return coin_history

    async def create_data_container_operations(self, *args, **kwargs):
        data_container = XArrayDataContainerOperations(*args, **kwargs)
        await data_container.initialize_container()
        return data_container


@register_factory("data")
class ConcreteSomeOtherFactory(DataFactory):
    def create_coin_history_obtainer(self, *args, **kwargs):
        return SomeOtherCoinHistoryObtainer(*args, **kwargs)

    def create_data_container_operations(self, *args, **kwargs):
        return SomeOtherDataContainerOperations(*args, **kwargs)


class AbstractCoinHistoryObtainer(ABC):
    def __init__(self, exchange_factory: StockMarketFactory,
                 interval,
                 start_str,
                 end_str,
                 limit,
                 ticker_pool=None):
        self.market_requester = exchange_factory.create_market_requester()
        self.market_operations = exchange_factory.create_market_operations(self.market_requester)
        self.market_harmonizer = exchange_factory.create_data_homogenizer(self.market_operations)
        self.data_container = None
        self.interval = interval
        self.start_str = start_str
        self.end_str = end_str
        self.limit = limit
        self.ticker_pool = ticker_pool
        self.example_raw_history = None

    async def initialize_example(self):
        self.example_raw_history = self.example_raw_history or list(await self._get_raw_history_for_ticker("ETHBTC"))
        return self.example_raw_history

    async def _get_raw_history_for_ticker(self, ticker_symbol: str):
        return await self.market_harmonizer.get_history_for_ticker(ticker=ticker_symbol,
                                                                   interval=self.interval,
                                                                   start_str=self.start_str,
                                                                   end_str=self.end_str,
                                                                   limit=self.limit)

    async def initialize_ticker_pool(self):
        return await self.market_harmonizer.get_all_coins_ticker_objects()


class XArrayCoinHistoryObtainer(AbstractCoinHistoryObtainer):
    async def get_depth_of_indices(self):
        example_history = await self.initialize_example()
        indices = list(range(len(list(example_history))))
        return indices

    async def get_coords_for_data_array(self):
        base_assets = self.get_set_of_ticker_attributes("baseAsset")
        reference_assets = self.get_set_of_ticker_attributes("quoteAsset")
        fields = self.market_harmonizer.HistoryFields._fields
        # TODO Use inheritance to avoid directly accessing private member
        return [list(base_assets),
                list(reference_assets),
                list(fields),
                await self.get_depth_of_indices()]

    def get_set_of_ticker_attributes(self, asset_type: str):
        fields = set()
        for ticker in self.ticker_pool:
            fields.add(getattr(ticker, asset_type))
        return fields

    async def get_historical_data_all_coins(self):
        tasks_to_pursue = {}
        for ticker in self.ticker_pool:
            tasks_to_pursue[ticker.baseAsset, ticker.quoteAsset] = \
                self._get_raw_history_for_ticker(ticker.symbol)
        return await general_utilities.gather_dict(tasks_to_pursue)


class SomeOtherCoinHistoryObtainer(AbstractCoinHistoryObtainer):
    def get_filled_container(self):
        pass


class AbstractDataContainerOperations(ABC):
    def __init__(self, history_obtainer):
        self.history_obtainer = history_obtainer
        self.data_container = None

    @abstractmethod
    async def get_filled_container(self):
        pass

    @abstractmethod
    def initialize_container(self):
        pass


class XArrayDataContainerOperations(AbstractDataContainerOperations):
    async def initialize_container(self):
        if self.data_container is None:
            coords = await self.history_obtainer.get_coords_for_data_array()
            dims = ["base_asset", "reference_asset", "item_to_compare", "index_number"]
            self.data_container = xr.DataArray(None, coords=coords, dims=dims)

    async def get_filled_container(self):
        await self.populate_container()
        return self.data_container

    async def populate_container(self):
        historical_data = await self.history_obtainer.get_historical_data_all_coins()
        for (base_asset, reference_asset), ticker_history in historical_data.items():
            try:
                history_df = await self.get_compatible_df(ticker_history)
            except exceptions.EmptyDataFrameException:
                continue
            self.data_container.loc[base_asset, reference_asset] = history_df.transpose()
            logger.debug(f"History set in x_array for ticker {base_asset}{reference_asset}")

    @staticmethod
    def calculate_rows_to_add(df, list_of_standard_history):
        df_rows, _ = df.shape
        expected_rows = len(list_of_standard_history)
        return expected_rows - df_rows

    async def get_compatible_df(self, ticker_history):
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
        new_indices_to_add = list(range(df.index[-1] + 1, df.index[-1] + 1 + empty_rows_to_add))
        return df.reindex(df.index.to_list() + new_indices_to_add)


class SomeOtherDataContainerOperations(AbstractDataContainerOperations):
    def initialize_container(self):
        pass

    def get_filled_container(self):
        pass
