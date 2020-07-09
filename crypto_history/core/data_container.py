from __future__ import annotations
import asyncio
import random
import logging
import pandas as pd
import numpy as np
import xarray as xr
from itertools import groupby
from abc import ABC, abstractmethod
from dataclasses import asdict
from .get_market_data import ConcreteBinanceFactory, StockMarketFactory
from collections import Iterator
from ..utilities import general_utilities, exceptions

logger = logging.getLogger(__name__)


class DataContainerFactory(ABC):
    @abstractmethod
    def create_container_symbol_info(self):
        pass

    def create_container_coin_history(self):
        pass


class ConcreteXArrayFactory(DataContainerFactory):
    def create_container_symbol_info(self, *args, **kwargs):
        return XArraySymbolInfo(*args, **kwargs)

    async def create_container_coin_history(self, *args, **kwargs):
        coin_history = XArrayCoinHistory(*args, **kwargs)
        coin_history.ticker_pool = await coin_history.initialize_ticker_pool()
        return coin_history


class ConcreteSomeOtherFactory(DataContainerFactory):
    def create_container_coin_history(self, *args, **kwargs):
        return SomeOtherSymbolInfo(*args, **kwargs)

    def create_container_symbol_info(self, *args, **kwargs):
        return SomeOtherCoinHistory(*args, **kwargs)


class AbstractSymbolInfo(ABC):
    pass


class XArraySymbolInfo(AbstractSymbolInfo):
    pass


class SomeOtherSymbolInfo(AbstractSymbolInfo):
    pass


class AbstractCoinHistory(ABC):
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

    @abstractmethod
    async def get_filled_container(self):
        pass

    async def initialize_example(self):
        self.example_raw_history = self.example_raw_history or list(await self._get_raw_history_for_ticker("ETHBTC"))

    async def _get_raw_history_for_ticker(self, ticker_symbol: str):
        return await self.market_harmonizer.get_history_for_ticker(ticker=ticker_symbol,
                                                                   interval=self.interval,
                                                                   start_str=self.start_str,
                                                                   end_str=self.end_str,
                                                                   limit=self.limit)

    async def initialize_ticker_pool(self):
        return await self.market_harmonizer.get_all_coins_ticker_objects()


class XArrayCoinHistory(AbstractCoinHistory):
    async def get_depth_of_indices(self):
        await self.initialize_example()
        indices = list(range(len(list(self.example_raw_history))))
        return indices

    async def get_coords_for_data_array(self):
        base_assets = self.get_set_of_ticker_attributes("baseAsset")
        reference_assets = self.get_set_of_ticker_attributes("quoteAsset")
        fields = self.market_harmonizer.History._fields
        # TODO Use inheritance to avoid directly accessing private member
        return [list(base_assets),
                list(reference_assets),
                list(fields),
                await self.get_depth_of_indices()]

    async def initialize_data_array(self):
        coords = await self.get_coords_for_data_array()
        return xr.DataArray(None, coords=coords)

    @staticmethod
    def add_extra_rows_to_bottom(df: pd.DataFrame, empty_rows_to_add: int):
        new_indices_to_add = list(range(df.index[-1] + 1, df.index[-1] + 1 + empty_rows_to_add))
        return df.reindex(df.index.to_list() + new_indices_to_add)


    @staticmethod
    def calculate_rows_to_add(df, list_of_standard_history):
        df_rows, _ = df.shape
        expected_rows = len(list_of_standard_history)
        return expected_rows - df_rows

    def get_compatible_df(self, ticker_history):
        # TODO Assuming that the df is only not filled in the bottom
        example_standard_history = self.example_raw_history
        history_df = pd.DataFrame(ticker_history)
        if history_df.empty:
            raise exceptions.EmptyDataFrameException
        rows_to_add = self.calculate_rows_to_add(history_df, example_standard_history)
        if rows_to_add > 0:
            history_df = self.add_extra_rows_to_bottom(history_df, rows_to_add)
        return history_df

    async def get_filled_container(self):
        xr_array = await self.initialize_data_array()
        historical_data = await self.get_historical_data()
        for (base_asset, reference_asset), ticker_history in historical_data.items():
            try:
                history_df = self.get_compatible_df(ticker_history)
            except exceptions.EmptyDataFrameException:
                continue
            xr_array.loc[base_asset, reference_asset] = history_df.transpose()
            logger.debug(f"History set in x_array for ticker {base_asset}{reference_asset}")
        return xr_array

    def get_set_of_ticker_attributes(self, asset_type: str):
        fields = set()
        for ticker in self.ticker_pool:
            fields.add(getattr(ticker, asset_type))
        return fields

    async def get_historical_data(self):
        tasks_to_pursue = {}
        for ticker in self.ticker_pool:
            tasks_to_pursue[ticker.baseAsset, ticker.quoteAsset] = \
                self._get_raw_history_for_ticker(ticker.symbol)
        return await general_utilities.gather_dict(tasks_to_pursue)


class SomeOtherCoinHistory(AbstractCoinHistory):
    pass

