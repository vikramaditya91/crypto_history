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
from ..utilities import general_utilities

logger = logging.getLogger(__package__)


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

    @abstractmethod
    async def get_filled_container(self):
        pass

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
        history_example = await self._get_raw_history_for_ticker("ETHBTC")
        indices = list(map(lambda x: x.open_ts, history_example))
        return indices

    async def get_coords_for_data_array(self):
        base_assets = self.get_set_of_ticker_fields("baseAsset")
        reference_assets = self.get_set_of_ticker_fields("quoteAsset")
        fields = self.market_harmonizer.History._fields
        # TODO Use inheritance to avoid directly accessing private member

        return [list(base_assets),
                list(reference_assets),
                list(fields),
                await self.get_depth_of_indices()]

    async def get_filled_container(self):
        coords = await self.get_coords_for_data_array()
        xr_array = xr.DataArray(None, coords=coords)
        # FixMe Potential bug when the arrays might not be of the same length
        historical_data = await self.get_historical_data()
        for (base_asset, reference_asset), ticker_history in historical_data.items():
            history_df = pd.DataFrame(ticker_history)
            if history_df.shape == xr_array.shape[:-3:-1]:
                xr_array.loc[base_asset, reference_asset] = history_df.transpose()
            elif history_df.shape == (0, 0):
                logger.warning(f"ticker: {base_asset}:{reference_asset} does not have a history")
            else:
                a = 1
        return xr_array

    def get_set_of_ticker_fields(self, asset_type: str):
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

