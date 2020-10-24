import logging
import datetime
import xarray as xr
import numpy as np
import pandas as pd
import contextlib
from typing import List, Dict, Union, Tuple, Type
from crypto_history.stock_market.stock_market_factory import StockMarketFactory
from crypto_history.data_container.data_container_pre import PrimitiveDataArrayOperations
from crypto_history.utilities.exceptions import EmptyDataFrameException
from crypto_history.utilities.general_utilities import TypeVarPlaceHolder

logger = logging.getLogger(__name__)


class TimeStampIndexedDataContainer:
    """Responsible for transforming the Primitive DataArray to a DataArray\
     indexed by the timestamp of choice
     with possibility to approximate for easier handling"""

    def __init__(
        self,
        primitive_full_data_container: PrimitiveDataArrayOperations,
        primitive_reference_data_container: PrimitiveDataArrayOperations,
        aggregate_coordinate_by: str,
    ):
        self.aggregate_coordinate_by = aggregate_coordinate_by
        self.primitive_full_data_container = primitive_full_data_container
        self.primitive_reference_data_container = (
            primitive_reference_data_container
        )
        self.ohlcv_coord_name = "ohlcv_fields"

    @classmethod
    @contextlib.asynccontextmanager
    async def create_time_stamp_indexed_data_container(
        cls,
        exchange_factory: StockMarketFactory,
        base_assets: List,
        reference_assets: List,
        reference_ticker: tuple,
        aggregate_coordinate_by: str,
        ohlcv_fields: List,
        weight: str,
        start_time: Union[str, datetime.datetime, int],
        end_time: Union[str, datetime.datetime, int],
    ):
        """
        The factory for creating the time stamp indexed data container
        Args:
            exchange_factory (StockMarketFactory): The exchange factory
            base_assets(List): List of base assets
            reference_assets(List): List of reference assets
            reference_ticker(tuple): ('xxx', 'yyy') where xxx is the \
                base_asset and yyy is the reference asset \
                for indexing the timestamp
            aggregate_coordinate_by(str): The direction in which the \
                coordinates should be aggregated by
            ohlcv_fields(List): list of ohlcv-fields necessary to capture
            weight(str): weight/interval of the kline/candle
            start_time(str/datetime.datetime/int): start time/date of \
                the candles
            end_time(str/datetime.datetime/int): end time/date of \
                the candles

        Yields:
            TimeStampIndexedDataContainer constructed with above details

        """
        async with PrimitiveDataArrayOperations.\
                create_primitive_data_array_operations(
                    exchange_factory,
                    base_assets,
                    reference_assets,
                    ohlcv_fields,
                    weight,
                    start_time,
                    end_time,
                ) as primtive_data_operations:
            reference_base, reference_quote = reference_ticker
            async with PrimitiveDataArrayOperations.\
                    create_primitive_data_array_operations(
                        exchange_factory,
                        [reference_base],
                        [reference_quote],
                        ohlcv_fields,
                        weight,
                        start_time,
                        end_time,
                    ) as primitive_reference_data_container:
                yield cls(
                    primtive_data_operations,
                    primitive_reference_data_container,
                    aggregate_coordinate_by,
                )

    @staticmethod
    async def get_primitive_xr_dataarray(
        data_container_object: PrimitiveDataArrayOperations,
    ) -> xr.DataArray:
        """
        Gets the primitive xr.DataArray from the data container object
        Returns:
            xr.DataArray: the actual xr.DataArray
        """
        logger.debug(
            f"Getting the primitive dataarray for {data_container_object}"
        )
        return await data_container_object.get_populated_primitive_container()

    async def get_primitive_reference_xr_dataarray(self) -> xr.DataArray:
        """
        Gets the reference data container
        Returns:
            xr.DataArray: the reference data container (eg. ETHBTC history)

        """
        logger.debug("Obtain the primitive reference dataarray")
        return await self.get_primitive_xr_dataarray(
            self.primitive_reference_data_container
        )

    async def get_primitive_full_xr_dataarray(self) -> xr.DataArray:
        """
        Gets the full data container
        Returns:
            xr.DataArray: the complete data container

        """
        logger.debug("Obtain the primitive complete dataarray")
        return await self.get_primitive_xr_dataarray(
            self.primitive_full_data_container
        )

    @staticmethod
    def get_all_unique_values(
        dataarray: xr.DataArray, coord_name: str, field_in_coord: str
    ) -> List:
        """
        Gets all the unique values in the xr.DataArray in the particular\
         item of the coordinate
        Args:
            dataarray (xr.DataArray): whose data is to be selected
            coord_name (str): name of the coordinate which is the \
            primary selector
            field_in_coord (str): name of the item in the coordinate\
             which is the second level of select

        Returns:
            list of all the items in the selected coordinate, \
            field sorted increasingly

        """
        selected_da = dataarray.sel(
            {coord_name: field_in_coord}
        )
        flattened_np_array = selected_da.values.flatten()
        none_removed = flattened_np_array[flattened_np_array != np.array(None)]
        nan_removed = none_removed[~pd.isnull(none_removed)]
        list_of_all_ts = list(set(nan_removed.tolist()))
        list_of_all_ts.sort()
        logger.debug(
            f"The unique values of {coord_name} of field {field_in_coord}"
            f" in the dataarray are {list_of_all_ts}"
        )
        return list_of_all_ts

    async def get_timestamps_of_reference_dataarray(
        self, field_in_coord: str
    ) -> List:
        """
        Obtains the time stamp list from the reference dataarray
        Args:
            field_in_coord (str): the item in the ohlcv_fields which\
             is of interest

        Returns:
            list of the time stamps from the reference dataarray
        """
        reference_dataarray = await self.get_primitive_reference_xr_dataarray()
        logger.debug("Getting the timestamp from the reference dataarray")
        return (
            reference_dataarray.sel(ohlcv_fields=field_in_coord)
            .values.flatten()
            .tolist()
        )

    async def get_timestamps_for_new_dataarray(
        self, do_approximation: bool, dataarray: xr.DataArray,
    ) -> List:
        """Obtains the time stamp for the new dataarray.
        It may either select it from the reference dataarray or\
         get it from the actual dataarray if approximation is not desired
        Args:
            do_approximation (bool): if True, timestamps are approximated and\
             taken from the reference dataarray. if False, timestamps are not\
              approximated. All timestamps from all various tickers will be\
               the coordinate for the timestamp
            dataarray (xr.DataArray): the original dataarray which contains\
             all the timestamps in it

        Returns:
             list of the timestamps for the new dataarray
        """
        if do_approximation is True:
            logger.info(
                "The timestamps are going to be approximated to the reference "
                "dataarray's timestamps"
            )
            return await self.get_timestamps_of_reference_dataarray(
                self.aggregate_coordinate_by
            )
        else:
            logger.info("The timestamps will not be approximated")
            return self.get_all_unique_values(
                dataarray, self.ohlcv_coord_name, self.aggregate_coordinate_by
            )

    async def get_xr_dataarray_indexed_by_timestamps(
        self, do_approximation: bool = True, tolerance_ratio: float = 0.001
    ) -> xr.DataArray:
        """
        Gets the xr.DataSet of the coin histories of the particular chunk.
        Obtains the data and then transforms it to the xr.DataSet
        Args:
            do_approximation (bool): check if the timestamps can be \
            approximated to the reference datarray
            tolerance_ratio (float): the ratio of the maximum tolerance for
            approximations of timestamps
        Returns:
            xr.DataSet data of the coin history
        """
        populated_dataarray = await self.get_primitive_full_xr_dataarray()
        new_df = await self.generate_empty_df_with_new_timestamps(
            do_approximation, populated_dataarray
        )

        tolerance = await self.get_value_of_tolerance(
            do_approximation=do_approximation, tolerance_ratio=tolerance_ratio
        )

        index_of_integrating_ts = self.get_index_of_field(
            populated_dataarray,
            self.ohlcv_coord_name,
            self.aggregate_coordinate_by,
        )

        return self._populate_ts_indexed_dataarray(
            populated_dataarray, new_df, index_of_integrating_ts, tolerance
        )

    async def generate_empty_df_with_new_timestamps(
        self, do_approximation: bool, old_dataarray: xr.DataArray
    ) -> xr.DataArray:
        """
        Generates an empty dataframe with new timestamps in the coordinates
        Args:
            do_approximation (bool): Check if the timestamps can be \
             approximated
            old_dataarray (xr.DataArray): The old dataarray used for reference
            for constructing new dataarray

        Returns:
            xr.DataArray The new empty dataarray which has the coordinates set

        """
        reference_ts = await self.get_timestamps_for_new_dataarray(
            do_approximation, old_dataarray
        )

        new_coordinates = self.get_coords_for_timestamp_indexed_datarray(
            old_dataarray, reference_ts
        )
        logger.debug(
            f"The new dataframe will be initialized with"
            f" coordinates: {new_coordinates}"
        )
        return self.generate_empty_dataarray_indexed_by_timestamp(
            coords=new_coordinates
        )

    async def get_value_of_tolerance(
        self, do_approximation: bool, tolerance_ratio: float = 0.001
    ):
        """
        Gets the value of tolerance used for reindexing
        Args:
            do_approximation (bool): Check if approximation can be made
            tolerance_ratio (float): The ratio of maximum tolerance of time \
            difference for generating \
            coordinates of timestamp

        Returns:
            float, value of the tolerance
        """
        if do_approximation is True:
            return await self.calculate_tolerance_value_from_ratio(
                tolerance_ratio
            )
        else:
            return 0

    def get_coords_for_timestamp_indexed_datarray(
        self,
        old_dataarray: xr.DataArray,
        reference_ts: List[int]
    ) -> Dict:
        """
        Gets the coordinates for the timestamp index dataarray
        Args:
            old_dataarray (xr.DataArray): The old dataarray taken as the\
             building block for the new dataarray
            reference_ts (list): The new timestamps which serve as the\
             references instead of index_number

        Returns:
            dictionary of coordinates for the new dataarray

        """
        index_of_integrating_ts = self.get_index_of_field(
            old_dataarray, self.ohlcv_coord_name, self.aggregate_coordinate_by
        )

        new_ohlcv_fields = old_dataarray[self.ohlcv_coord_name].values.tolist()
        new_ohlcv_fields.pop(index_of_integrating_ts)
        coords = {
            "base_assets": old_dataarray.base_assets,
            "reference_assets": old_dataarray.reference_assets,
            "timestamp": reference_ts,
            "ohlcv_fields": new_ohlcv_fields,
        }
        return coords

    @staticmethod
    def get_index_of_field(
        data_array: xr.DataArray, coord_name: str, item_in_coord: str
    ) -> int:
        """
        Gets the index of the particular field in the coordinate
        Args:
            data_array (xr.DataArray): in which the index is to be identified
            coord_name (str): name of the coordinate
            item_in_coord (str): name of the field in the coordinate

        Returns:
            int, the index number which can be used to identify the item

        """
        return data_array[coord_name].values.tolist().index(item_in_coord)

    @staticmethod
    def generate_empty_dataarray_indexed_by_timestamp(
        coords: Dict,
    ) -> xr.DataArray:
        """
        Generates the new dataarray from the coordinates provided
        Args:
            coords(Dict): the coordinates for the new dataarray

        Returns:
            xr.DataArray the new empty dataarray

        """
        return xr.DataArray(None, coords=coords, dims=coords.keys())

    async def calculate_tolerance_value_from_ratio(
        self, ratio: float = 0.001
    ) -> float:
        """
        Calculates the tolerance value based on the tolerance ratio
        Args:
            ratio (float): ratio of the tolerance

        Returns:
            float, value of the tolerance value

        """
        standard_ts = await self.get_timestamps_of_reference_dataarray(
            self.aggregate_coordinate_by
        )
        standard_diff = np.diff(standard_ts).mean()
        logger.debug(
            f"The standard difference between timestamps while calculating"
            f" timestamps are {standard_diff}"
        )
        return standard_diff * ratio

    @staticmethod
    def get_df_to_insert(
        sub_dataarray: xr.DataArray,
        index_of_integrating_ts: int,
        reference_ts: List,
        tolerance: float,
    ) -> pd.DataFrame:
        """
        Calculates the dataframe from the dataarray with one reduced shape size
        Args:
            sub_dataarray (xr.DataArray): the sub dataarray whose dataframe \
            is to be extracted
            index_of_integrating_ts (int): index of the integrating field \
            in the coordinate
            reference_ts (List): list of the indexes according to what the \
            dataframe needs to be adjusted
            tolerance (float): the value of the tolerance which can be \
            considered while matching indices

        Returns:
            pd.DataFrame the reindexed dataframe which contains the data \
            in the required indices

        Raises:
            EmptyDataFrameException if the sub_dataarray contains\
             only na values
        """
        df = pd.DataFrame(sub_dataarray.values.transpose())
        if not df.isna().all().all():
            df = df.dropna()
            df = df.set_index(index_of_integrating_ts)
            df = df.reindex(
                reference_ts, method="nearest", tolerance=tolerance
            )
            df.sort_index()
            return df
        raise EmptyDataFrameException(
            "The data only contains na values. \n"
            "Dataframe cannot be successfully generated"
        )

    def _populate_ts_indexed_dataarray(
        self,
        old_dataarray: xr.DataArray,
        new_dataarray: xr.DataArray,
        index_of_integrating_ts: int,
        tolerance: float,
    ) -> xr.DataArray:
        """
        Populates the new dataarray with the new indexes
        Args:
            old_dataarray (xr.DataArray): The old dataarray which contains the\
            values to build the new dataframe
            new_dataarray: (xr.DataArray): The new empty dataarray whose\
             coordinates have already been set
            index_of_integrating_ts (int): The index of the integrating ts.\
             Essentially, the index of open_ts in ohlcv_fields
            tolerance: the value of the tolerance for reindexing

        Returns:
            xr.DataArray the dataarray whose indices are of the timestamp\
             and the data is filled

        """
        reference_ts = new_dataarray.timestamp.values.tolist()
        # FixMe This is probably not the quickest way to get a new dataarray.
        # Particularly, for non-approximated method should be reindixable?
        for base_coin_iter in old_dataarray:
            for ref_coin_iter in base_coin_iter:
                try:
                    df_to_insert = self.get_df_to_insert(
                        ref_coin_iter,
                        index_of_integrating_ts,
                        reference_ts,
                        tolerance,
                    )
                    new_dataarray.loc[
                        ref_coin_iter.base_assets.values.tolist(),
                        ref_coin_iter.reference_assets.values.tolist(),
                    ] = df_to_insert
                    logger.debug(
                        "The dataframe has been inserted in"
                        " the new dataframe for"
                        f"{ref_coin_iter.base_assets.values.tolist()}"
                        f"{ref_coin_iter.reference_assets.values.tolist()}."
                    )
                except EmptyDataFrameException:
                    logger.debug(
                        f"Empty dataframe for combination of "
                        f"{ref_coin_iter.base_assets.values.tolist()}"
                        f"{ref_coin_iter.reference_assets.values.tolist()}."
                        f" Skipping the df"
                    )
        return new_dataarray


class TimeAggregatedDataContainer:
    """Aggregates various time-ranges into the data container"""
    def __init__(
        self,
        exchange_factory: StockMarketFactory,
        base_assets: List[str],
        reference_assets: List[str],
        ohlcv_fields: List[str],
        time_range_dict: Dict[Tuple[Union[str, datetime.datetime],
                                    Union[str, datetime.datetime]],
                              str],
        reference_ticker: Tuple = ("ETH", "BTC"),
        aggregate_coordinate_by: str = "open_ts",
    ):
        self.exchange_factory = exchange_factory
        self.base_assets = base_assets
        self.reference_assets = reference_assets
        self.ohlcv_fields = ohlcv_fields
        self.time_range_dict = time_range_dict
        self.reference_ticker = reference_ticker
        self.aggregate_coordinate_by = aggregate_coordinate_by
        self.time_interval_splitter = (
            exchange_factory.create_time_interval_chunks()
        )

    @classmethod
    def create_instance(cls: Type[TypeVarPlaceHolder],
                        exchange_factory: StockMarketFactory,
                        base_assets: List[str],
                        reference_assets: List[str],
                        ohlcv_fields: List[str],
                        time_range_dict:
                        Dict[Tuple[Union[datetime.timedelta, str],
                                   Union[datetime.timedelta, str]],
                             str],
                        reference_ticker: Tuple = ("ETH", "BTC"),
                        aggregate_coordinate_by: str = "open_ts",
                        ) -> TypeVarPlaceHolder:
        """
        Creates the time-aggregator from time-deltas which go back from the current time
        Args:
            exchange_factory (StockMarketFactory): The exchange factory
            base_assets (List): base-asset coins 
            reference_assets (List): reference-asset coins 
            ohlcv_fields (List): ohlcv-fields to calculate 
            time_range_dict (Dict): dictionary of time-ranges where the keys are \
            tuples of timedeltas 
            reference_ticker (Tuple): reference-ticker for reference timestamps 
            aggregate_coordinate_by (str): Timestamp indexed by this value 

        Returns:
            TimeAggregatedDataContainer

        """
        if all(map(
                lambda x:isinstance(x[0], datetime.timedelta)
                         and
                         isinstance(x[1], datetime.timedelta),
                time_range_dict.keys())):
            time_now = datetime.datetime.now()
            time_range_temp_dict = {}
            for (time_start, time_end), candle in time_range_dict.items():
                datetime_start = time_now - time_start
                datetime_end = time_now - time_end
                time_range_temp_dict[(datetime_start,
                                      datetime_end)] = candle
                time_range_dict = time_range_temp_dict
        elif all(map(
                lambda x:isinstance(x[0], str)
                         and
                         isinstance(x[1], str),
                time_range_dict.keys())):
            pass
        else:
            raise TypeError("Invalid time values")
        return cls(
            exchange_factory,
            base_assets=base_assets,
            reference_assets=reference_assets,
            ohlcv_fields=ohlcv_fields,
            time_range_dict=time_range_dict,
            reference_ticker=reference_ticker,
            aggregate_coordinate_by=aggregate_coordinate_by
        )

    def get_time_interval_chunks(self, time_range: Dict) -> List[tuple]:
        """
        Gets the time interval chunks from the raw dict provided
        Args:
            time_range: Dict with key=tuple of start and end time\
                                  value=type of klines/intervals

        Returns:
            List of tuples with the individual requests for histories
                     ((start time, end time), interval)

        """
        return self.time_interval_splitter.get_time_range_for_historical_calls(
            time_range
        )

    @staticmethod
    def concatenate_dataarray_in_coord(
        *dataarray: Union[List[xr.DataArray], Tuple[xr.DataArray]],
        coordinate="timestamp",
    ) -> xr.DataArray:
        """
        Concatenates two histories in the desired coordinate/dimension
        Args:
            dataarray List[xr.DataArray]: List/Tuple of xr.DataArrays \
                that need to be concatednated
            coordinate: coordinate direction to concat by

        Returns:
            Concatenated xr.DataArray consisting of all the data concatenated

        """
        return xr.concat(
            list(filter(lambda x: x is not None, dataarray)), dim=coordinate
        )

    async def get_chunk_history(
        self,
        kline_width: str,
        start_time: Union[str, datetime.datetime, int],
        end_time: Union[str, datetime.datetime, int],
    ) -> xr.DataArray:
        """
        Gets the history of the particular chunk whose start and end
        times are well defined
        Args:
            kline_width (str): The exchanges interval
            start_time (datetime/str/int): the start-time of the \
                        chunk of history
            end_time: (datetime/str/int): the end-time of the \
                        chunk of history

        Returns:
            xr.DataArray the history of the chunk

        """
        async with TimeStampIndexedDataContainer.\
                create_time_stamp_indexed_data_container(
                    exchange_factory=self.exchange_factory,
                    base_assets=self.base_assets,
                    reference_assets=self.reference_assets,
                    reference_ticker=self.reference_ticker,
                    aggregate_coordinate_by=self.aggregate_coordinate_by,
                    ohlcv_fields=self.ohlcv_fields,
                    weight=kline_width,
                    start_time=start_time,
                    end_time=end_time,
                ) as time_stamp_indexed_container:
            return await time_stamp_indexed_container.\
                get_xr_dataarray_indexed_by_timestamps(
                    do_approximation=False
                )

    async def get_time_aggregated_data_container(
        self, sort: bool = True
    ) -> xr.DataArray:
        """
        Gets time aggregated data container by splitting
         the containers as required

        Args:
            sort (bool): to check if the data is to be sorted \
            in an increasing order

        Returns:
            xr.DataArray The complete history of the interval

        """
        chunks_of_time = self.get_time_interval_chunks(self.time_range_dict)
        chunks = []
        for (start_time, end_time), interval in chunks_of_time:
            chunk_history = await self.get_chunk_history(
                interval, start_time, end_time
            )
            chunks.append(chunk_history)
        overall_history = self.concatenate_dataarray_in_coord(*chunks)
        if sort is True:
            overall_history = self.get_sorted_dataarray(
                overall_history, "timestamp"
            )
        return overall_history

    @staticmethod
    def get_sorted_dataarray(
        dataarray: xr.DataArray,
        coordinate: str
    ) -> xr.DataArray:
        """
        Sorts the given dataarray in the given coordinate direction
        Args:
            dataarray (xr.DataArray): The dataarray which has to be sorted
            coordinate: sort the dataarray in the particular \
                direction of the coordinate

        Returns:
            xr.DataArray The sorted dataarray

        """
        return dataarray.sortby(coordinate, ascending=True)
