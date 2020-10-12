from __future__ import annotations
import logging
import contextlib
import xarray as xr
from pandas import DataFrame
from dataclasses import dataclass, fields
from typing import Union, List, Dict
from datetime import datetime
from crypto_history.stock_market.tickers import TickerPool
from crypto_history.stock_market.stock_market_factory import (
    StockMarketFactory,
    AbstractMarketHomogenizer
)
from crypto_history.data_container.utilities import DataFrameOperations
from crypto_history.utilities import general_utilities, exceptions

logger = logging.getLogger(__name__)


class PrimitiveCoinHistoryObtainer:
    """Abstract class to serve as the parent to generating \
    histories obtainer classes"""

    def __init__(
        self,
        market_harmonizer: AbstractMarketHomogenizer,
        interval: str,
        start_time: Union[str, datetime, int],
        end_time: Union[str, datetime, int],
    ):
        self.market_harmonizer = market_harmonizer
        self.data_container = None
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.ticker_pool = None
        self.example_raw_history = None

    @classmethod
    @contextlib.asynccontextmanager
    async def create_primitive_coin_history_obtainer(
        cls,
        exchange_factory: StockMarketFactory,
        interval: str,
        start_time: Union[str, datetime, int],
        end_time: Union[str, datetime, int],
    ) -> PrimitiveCoinHistoryObtainer:
        """
        Generates the coin history obtainer

        Args:
            exchange_factory(StockMarketFactory): instance of\
             the exchange_factory which is responsible for \
            setting the market_homogenizer
            interval(str): Length of the history of the klines per item
            start_time(str/datetime/int): duration from which history \
               is necessary
            end_time(str/datetime/int): duration up to which history \
               is necessary

        Yields:
            PrimitiveCoinHistoryObtainer instance from the above \
            arguments
        """
        async with exchange_factory.create_data_homogenizer() \
                as market_harmonizer:
            yield cls(
                market_harmonizer, interval, start_time, end_time,
            )

    async def initialize_example(self) -> List:
        # FixMe The TimeIndexedDataContainer something similar.\
        #  Should be centralized?
        """
        Initializes an example of the raw history and stores\
         it example_raw_history

        Returns:
             list: instance of an example of a raw history of ETHvsBTC

        """
        self.example_raw_history = self.example_raw_history or list(
            await self._get_raw_history_for_ticker("ETHBTC")
        )
        return self.example_raw_history

    async def _get_raw_history_for_ticker(self, ticker_symbol: str) -> map:
        """
        Gets the raw history for the ticker by using the interface\
         to the market provided by the market homogenizer

        Args:
            ticker_symbol(str): Symbol of the ticker whose history\
             is desired

        Returns:
             map: raw mapped history of the ticker mapped to the \
             OHLCVFields instance

        """
        return await self.market_harmonizer.get_history_for_ticker(
            ticker=ticker_symbol,
            interval=self.interval,
            start_time=self.start_time,
            end_time=self.end_time,
        )

    async def get_all_raw_tickers(self) -> TickerPool:
        """
        Obtains the ticker pool by accessing the market homogenizer

        Returns:
             TickerPool: The TickerPool filled with all the ticker\
              available to the market/exchange

        """
        return await self.market_harmonizer.get_all_raw_tickers()

    async def get_historical_data_from_base_and_reference_assets(
        self, base_assets: List, reference_assets: List
    ) -> Dict:
        """
        Get the historical data for the combination of base\
         and reference assets
        Args:
            base_assets (List['str']): list of all base assets
            reference_assets (List['str']): list of all reference assets

        Returns:
            The kline historical data of the combinations
        """
        raw_ticker_pool = await self.get_all_raw_tickers()
        tickers_to_capture = []
        for base_asset in base_assets:
            for reference_asset in reference_assets:
                if any(
                    f"{base_asset}{reference_asset}" == raw_ticker["symbol"]
                    for raw_ticker in raw_ticker_pool
                ):
                    tickers_to_capture.append((base_asset, reference_asset))
        return await self._get_historical_data_relevant_coins(
            tickers_to_capture
        )

    async def _get_historical_data_relevant_coins(
        self, tickers_to_capture: List[tuple]
    ) -> Dict:
        """
        Get the history of all the tickers in the market/exchange

        Args:
            tickers_to_capture (list): List of tickers to capture

        Returns:
             map: map of all the histories in the exchange/market
             mapped from history to the OHLCVFields

        """
        tasks_to_pursue = {}
        for base_asset, reference_asset in tickers_to_capture:
            tasks_to_pursue[
                base_asset, reference_asset
            ] = self._get_raw_history_for_ticker(
                f"{base_asset}{reference_asset}"
            )
        return await general_utilities.gather_dict(tasks_to_pursue)


class PrimitiveDimensionsManager:
    """Class responsible for managing the dimensions/coordinates \
    of the data container"""

    @dataclass
    class Dimensions:
        """Class for keeping track of the dimensions of the XDataArray"""

        base_assets: List
        reference_assets: List
        ohlcv_fields: List
        index_number: List

    def __init__(self, history_obtainer):
        self.coin_history_obtainer = history_obtainer

    @classmethod
    def create_primitive_dimensions_manager(
        cls, history_obtainer: PrimitiveCoinHistoryObtainer
    ) -> PrimitiveDimensionsManager:
        """Initializes the dimensions manager with the coin_history_obtainer"""
        return PrimitiveDimensionsManager(history_obtainer)

    async def get_depth_of_indices(self) -> List:
        """
        Obtains the depth of the indices to identify how many time-stamp values
         each history has available

        Returns:
             list: The list of indices from 0..n-1
                     where n corresponds to the number of time-stamps\
                      in the history

        """
        example_history = await self.coin_history_obtainer.initialize_example()
        indices = list(range(len(list(example_history))))
        return indices

    @staticmethod
    async def set_coords_dimensions_in_empty_container(
        dataclass_dimensions_coordinates,
    ) -> xr.DataArray:
        """
        Initialize the xarray container with None values but\
         with the coordinates as it helps in the memory allocation
        Args:
            dataclass_dimensions_coordinates: dataclass of \
            coordinates/dimensions as the framework for generating \
            the XDataArray
        """
        dimensions = [
            dimension.name
            for dimension in fields(dataclass_dimensions_coordinates)
        ]
        coordinates = [
            getattr(dataclass_dimensions_coordinates, item)
            for item in dimensions
        ]
        return xr.DataArray(None, coords=coordinates, dims=dimensions)

    async def get_dimension_coordinates_from_fields_and_assets(
        self, ohlcv_fields: List, base_assets: List, reference_assets: List
    ):
        """
        Gets the dataclass containing the coordinates required \
        for creating the xr.DataArray
        Args:
            ohlcv_fields: fields related to open_ts, close_ts, and so on
            base_assets: list of base coins
            reference_assets: list of reference coins

        Returns:
            DataClass of the coordinates necessary to build the DataArray
        """
        index_number = await self.get_depth_of_indices()
        return self.Dimensions(
            base_assets=base_assets,
            reference_assets=reference_assets,
            ohlcv_fields=ohlcv_fields + ["weight"],
            index_number=index_number,
        )


class PrimitiveDataArrayOperations:
    """Abstract Base Class for generating the data operations"""

    def __init__(
        self,
        history_obtainer: PrimitiveCoinHistoryObtainer,
        dimensions_manager: PrimitiveDimensionsManager,
        base_assets: List,
        reference_assets: List,
        ohlcv_fields: List,
        interval: str,
    ):
        self.history_obtainer = history_obtainer
        self.dimension_manager = dimensions_manager
        self.dataframe_operations = DataFrameOperations()
        self.base_assets = base_assets
        self.reference_assets = reference_assets
        self.ohlcv_fields = ohlcv_fields
        self.interval = interval

    @classmethod
    @contextlib.asynccontextmanager
    async def create_primitive_data_array_operations(
        cls,
        exchange_factory: StockMarketFactory,
        base_assets: List,
        reference_assets: List,
        ohlcv_fields: List,
        interval: str,
        start_time: Union[str, datetime, int],
        end_time: Union[str, datetime, int],
    ):
        """
        Initializes the DataContainerOperations which is the user \
        end-point for the basic data building
        Args:
            exchange_factory (StockMarketFactory): factory of the exchange
            base_assets (List): list of base coins to be accumulated
            reference_assets (List): list of reference/quote-against\
             coins to be accumulated
            ohlcv_fields (List): list of fields for the various fields
            interval (str): data capture interval
            start_time (str/datetime/int): date from which data collection \
                should start
            end_time (str/datetime/int): date up to which data collection \
                should be made
        Yields:
            PrimitiveDataArrayOperations instance from the above arguments
        """
        async with PrimitiveCoinHistoryObtainer.\
                create_primitive_coin_history_obtainer(
                exchange_factory, interval, start_time, end_time
                ) as history_obtainer:
            dimensions_manager = PrimitiveDimensionsManager.\
                create_primitive_dimensions_manager(
                    history_obtainer
                )
            yield cls(
                history_obtainer,
                dimensions_manager,
                base_assets,
                reference_assets,
                ohlcv_fields,
                interval,
            )

    async def get_populated_primitive_container(self) -> xr.DataArray:
        """
        Populates the container and returns it to the use

        Returns:
             xr.DataArray: the filled container of information

        """
        # TODO Avoid populating it every time it is called
        coord_dimension_dataclass = await self.dimension_manager.\
            get_dimension_coordinates_from_fields_and_assets(
                ohlcv_fields=self.ohlcv_fields,
                base_assets=self.base_assets,
                reference_assets=self.reference_assets,
            )

        data_container = await self.dimension_manager.\
            set_coords_dimensions_in_empty_container(
                coord_dimension_dataclass
            )
        data_container = await self.populate_container(
            data_container, coord_dimension_dataclass
        )
        return data_container

    @staticmethod
    def _insert_coin_history_in_container(
        data_container, base_asset, reference_asset, history_df
    ):
        """
        Low level function to inserts the coin history in the df
        Args:
            base_asset (str): base asset of the coin
            reference_asset (str): reference asset of the coin
            history_df (pandas.DataFrame): data frame of the coin history

        Returns:
            None
        """
        logger.debug(
            f"History set in x_array for ticker {base_asset}{reference_asset}"
        )
        data_container.loc[
            base_asset, reference_asset
        ] = history_df.transpose()
        return data_container

    @staticmethod
    def append_column_to_df(df, column_name, column_value) -> DataFrame:
        """
        Appends a new column to a df with constant value
        Args:
            df (DataFrame): dataframe to which column is to be appended
            column_name(str): name of the column
            column_value(str): value of the column

        Returns:
            new column added DataFrame
        """
        df[column_name] = column_value
        return df

    async def get_example_history(self) -> List:
        """A standard example on which the gaps are padded"""
        await self.history_obtainer.initialize_example()
        return self.history_obtainer.example_raw_history

    async def populate_container(
        self, data_container: xr.DataArray, coord_dimension_dataclass
    ) -> xr.DataArray:
        """
        Populates the xarray.DataArray with the historical \
        data of all the coins
        """
        historical_data = await self.history_obtainer.\
            get_historical_data_from_base_and_reference_assets(
                base_assets=coord_dimension_dataclass.base_assets,
                reference_assets=coord_dimension_dataclass.reference_assets,
            )
        standard_example_df = await self.get_example_history()
        for (
            (base_asset, reference_asset),
            ticker_raw_history,
        ) in historical_data.items():
            try:
                history_df = await self.dataframe_operations.get_compatible_df(
                    standard_example_df, ticker_raw_history
                )
            except exceptions.EmptyDataFrameException:
                logger.debug(
                    f"History does not exist for the combination "
                    f"of {base_asset}{reference_asset}"
                )
                continue
            else:
                history_df = self.dataframe_operations.\
                    drop_unnecessary_columns_from_df(
                        history_df, coord_dimension_dataclass.ohlcv_fields
                    )
                history_df = self.append_column_to_df(
                    history_df, "weight", self.interval
                )
            data_container = self._insert_coin_history_in_container(
                data_container, base_asset, reference_asset, history_df
            )
        return data_container
