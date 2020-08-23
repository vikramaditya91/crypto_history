from .primitive_data_builder import PrimitiveDataArrayOperations
import logging
import xarray as xr
import numpy as np
import pandas as pd
from typing import List, Dict
from ..utilities.exceptions import EmptyDataFrameException


logger = logging.getLogger(__name__)


class TimeStampIndexedDataContainer:
    """Responsible for transforming the DataArray provided by the primite_data_builder to a more usable DataSet"""
    def __init__(self,
                 exchange_factory,
                 base_assets,
                 reference_assets,
                 reference_ticker,
                 aggregate_coordinate_by,
                 ohlcv_fields,
                 weight,
                 start_str,
                 end_str,
                 limit):
        self.aggregate_coordinate_by = aggregate_coordinate_by
        self.primitive_full_data_container = PrimitiveDataArrayOperations(exchange_factory,
                                                                          base_assets,
                                                                          reference_assets,
                                                                          ohlcv_fields,
                                                                          weight,
                                                                          start_str,
                                                                          end_str,
                                                                          limit)
        self.reference_base, self.reference_quote = reference_ticker
        self.primitive_reference_data_container = PrimitiveDataArrayOperations(exchange_factory,
                                                                               [self.reference_base],
                                                                               [self.reference_quote],
                                                                               ohlcv_fields,
                                                                               weight,
                                                                               start_str,
                                                                               end_str,
                                                                               limit)
        self.ohlcv_coord_name = "ohlcv_fields"

    @staticmethod
    async def get_primitive_xr_dataarray(data_container_object: PrimitiveDataArrayOperations) -> xr.DataArray:
        """
        Gets the primitive xr.DataArray container
        Returns:
            xr.DataArray: the actual xr.DataArray
        """
        logger.info(f"Getting the primitive dataarray for {data_container_object}")
        return await data_container_object.get_populated_primitive_container()

    async def get_primitive_reference_xr_dataarray(self) -> xr.DataArray:
        """
        Gets the reference data container
        Returns:
            xr.DataArray: the reference data container (eg. ETHBTC history)

        """
        logger.info("Obtain the primitive reference dataarray")
        return await self.get_primitive_xr_dataarray(self.primitive_reference_data_container)

    @staticmethod
    def get_all_unique_values(dataarray: xr.DataArray,
                              coord_name: str,
                              field_in_coord: str) -> List:
        """
        Gets all the unique values in the xr.DataArray in the particular coordinate in the particular\
         item in the coordinate
        Args:
            dataarray: dataarray whose data is to be selected
            coord_name: name of the coordinate which is the primary selector
            field_in_coord: name of the item in the coordinate which is the second level of select

        Returns:
            list of all the items in the selected coordinate, field sorted increasingly

        """
        selected_da = dataarray.sel({coord_name:field_in_coord})
        list_of_all_ts = list(set(selected_da.values.flatten().tolist()))
        if None in list_of_all_ts:
            list_of_all_ts.remove(None)
        list_of_all_ts.sort()
        logger.info(f"The unique values of {coord_name} of field {field_in_coord}"
                    f" in the dataarray are {list_of_all_ts}")
        return list_of_all_ts

    async def get_ts_of_reference_da(self,
                                     field_in_coord: str) -> List:
        """
        Obtains the time stamp list from the reference dataarray
        Args:
            field_in_coord (str): the item in the ohlcv_fields which is of interest

        Returns:
            list of the time stamps from the reference dataarray
        """
        reference_dataarray = await self.get_primitive_reference_xr_dataarray()
        logger.info("Getting the timestamp from the reference dataarray")
        return reference_dataarray.sel(ohlcv_fields=field_in_coord).values.flatten().tolist()

    async def get_ts_for_new_dataarray(self,
                                       do_approximation: bool,
                                       dataarray: xr.DataArray,
                                       ) -> List:
        """Obtains the time stamp for the new dataarray.
        It may either select it from the reference dataarray or get it from the actual
        dataarray if approximation is not desired
        Args:
            do_approximation (bool): if True, timestamps are approximated and taken from the reference dataarray\
                                           if False, timestamps are not approximated. All timestamps from all various\
                                            tickers will be the coordinate for the timestamp
            dataarray (xr.DataArray): the original dataarray which contains all the timestamps in it

        Returns:
             list of the timestamps for the new dataarray
        """
        if do_approximation is True:
            logger.info("The timestamps are going to be approximated to the reference dataarray's timestamps")
            return await self.get_ts_of_reference_da(self.aggregate_coordinate_by)
        else:
            logger.info("The timestamps will not be approximated")
            return self.get_all_unique_values(dataarray, self.ohlcv_coord_name, self.aggregate_coordinate_by)

    async def get_xr_dataarray_ts_indexed(self,
                                          do_approximation=True,
                                          tolerance_ratio=0.001) -> xr.DataArray:
        """
        Gets the xr.DataSet of the coin histories of the particular chunk.
        Obtains the data and then transforms it to the xr.DataSet
        Args:
            do_approximation (bool): check if the timestamps can be approximated to the reference datarray
            tolerance_ratio (float): the ratio of the maximum tolerance for approximations of timestamps
        Returns:
            xr.DataSet data of the coin history
        """
        populated_dataarray = await self.get_primitive_xr_dataarray(self.primitive_full_data_container)
        populated_dataarray = self.damage_value_of_dataarray(populated_dataarray)

        new_df = await self.generate_empty_df_with_new_timestamps(do_approximation,
                                                                  populated_dataarray)

        tolerance = await self.get_value_of_tolerance(do_approximation=do_approximation,
                                                      tolerance_ratio=tolerance_ratio)

        index_of_integrating_ts = self.get_index_of_field(populated_dataarray,
                                                          self.ohlcv_coord_name,
                                                          self.aggregate_coordinate_by)

        return self.populate_new_dataarray(populated_dataarray,
                                           new_df,
                                           index_of_integrating_ts,
                                           tolerance)

    async def generate_empty_df_with_new_timestamps(self,
                                                    do_approximation: bool,
                                                    old_dataarray: xr.DataArray) -> xr.DataArray:
        """
        Generates an empty dataframe with new timestamps in the coordinates
        Args:
            do_approximation (bool): Check if the timestamps can be approximated 
            old_dataarray (xr.DataArray): The old dataarray used for reference for constructing new dataarray 

        Returns:
            xr.DataArray The new empty dataarray which has the coordinates set

        """
        reference_ts = await self.get_ts_for_new_dataarray(do_approximation,
                                                           old_dataarray)

        new_coordinates = self.get_coords_for_timestamp_indexed_datarray(old_dataarray,
                                                                         reference_ts)
        logger.info(f"The new dataframe will be initialized with coordinates: {new_coordinates}")
        return self.generate_empty_dataarray_indexed_by_timestamp(coords=new_coordinates)

    async def get_value_of_tolerance(self,
                                     do_approximation: bool,
                                     tolerance_ratio: float = 0.001):
        """
        Gets the value of tolerance used for reindexing
        Args:
            do_approximation (bool): Check if approximation can be made
            tolerance_ratio (float): The ratio of maximum tolerance of time difference for generating \
            coordinates of timestamp

        Returns:
            float, value of the tolerance
        """
        if do_approximation is True:
            return await self.calculate_tolerance_value_from_ratio(tolerance_ratio)
        else:
            return 0

    def get_coords_for_timestamp_indexed_datarray(self,
                                                  old_dataarray: xr.DataArray,
                                                  reference_ts: List) -> Dict:
        """
        Gets the coordinates for the timestamp index dataarray
        Args:
            old_dataarray (xr.DataArray): The old dataarray taken as the building block for the new dataarray
            reference_ts (list): The new timestamps which serve as the references instead of index_number

        Returns:
            dictionary of coordinates for the new dataarray

        """
        index_of_integrating_ts = self.get_index_of_field(old_dataarray,
                                                          self.ohlcv_coord_name,
                                                          self.aggregate_coordinate_by)

        new_ohlcv_fields = old_dataarray[self.ohlcv_coord_name].values.tolist()
        new_ohlcv_fields.pop(index_of_integrating_ts)
        coords = {'base_assets': old_dataarray.base_assets,
                  "reference_assets": old_dataarray.reference_assets,
                  "timestamp": reference_ts,
                  "ohlcv_fields": new_ohlcv_fields}
        return coords

    @staticmethod
    def get_index_of_field(data_array: xr.DataArray,
                           coord_name: str,
                           item_in_coord: str) -> int:
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
    def generate_empty_dataarray_indexed_by_timestamp(coords: Dict) -> xr.DataArray:
        """
        Generates the new dataarray from the coordinates provided
        Args:
            coords(Dict): the coordinates for the new dataarray

        Returns:
            xr.DataArray the new empty dataarray

        """
        return xr.DataArray(None, coords=coords, dims=coords.keys())

    async def calculate_tolerance_value_from_ratio(self,
                                                   ratio: float = 0.001) -> float:
        """
        Calculates the tolerance value based on the tolerance ratio
        Args:
            ratio (float): ratio of the tolerance

        Returns:
            float, value of the tolerance value

        """
        standard_ts = await self.get_ts_of_reference_da(self.aggregate_coordinate_by)
        standard_diff = np.diff(standard_ts).mean()
        logger.info(f"The standard difference between timestamps while calculating timestamps are {standard_diff}")
        return standard_diff*ratio

    @staticmethod
    def get_df_to_insert(sub_dataarray: xr.DataArray,
                         index_of_integrating_ts: int,
                         reference_ts: List,
                         tolerance: float) -> pd.DataFrame:
        """
        Calculates the dataframe from the dataarray with one reduced shape size
        Args:
            sub_dataarray (xr.DataArray): the sub dataarray whose dataframe is to be extracted
            index_of_integrating_ts (int): index of the integrating field in the coordinate
            reference_ts (List): list of the indexes according to what the dataframe needs to be adjusted
            tolerance (float): the value of the tolerance which can be considered while matching indices

        Returns:
            pd.DataFrame the reindexed dataframe which contains the data in the required indices

        Raises:
            EmptyDataFrameException if the sub_dataarray contains only na values
        """
        df = pd.DataFrame(sub_dataarray.values.transpose())
        if not df.isna().all().all():
            df = df.set_index(index_of_integrating_ts)
            return df.reindex(reference_ts, method="nearest", tolerance=tolerance)
        raise EmptyDataFrameException("The data only contains na values. \n"
                                      "Dataframe cannot be successfully generated")

    def populate_new_dataarray(self,
                               old_dataarray: xr.DataArray,
                               new_dataarray: xr.DataArray,
                               index_of_integrating_ts: int,
                               tolerance: float) -> xr.DataArray:
        """
        Populates the new dataarray with the new indexes
        Args:
            old_dataarray (xr.DataArray): The old dataarray which contains the values to build the new dataframe
            new_dataarray: (xr.DataArray): The new empty dataarray whose coordinates have already been set
            index_of_integrating_ts (int): The index of the integrating ts. Essentially, the index \
            of open_ts in ohlcv_fields
            tolerance: the value of the tolerance for reindexing

        Returns:
            xr.DataArray the dataarray whose indices are of the timestamp and the data is filled

        """
        reference_ts = new_dataarray.timestamp.values.tolist()
        for base_coin_iter in old_dataarray:
            for ref_coin_iter in base_coin_iter:
                try:
                    df_to_insert = self.get_df_to_insert(ref_coin_iter,
                                                         index_of_integrating_ts,
                                                         reference_ts,
                                                         tolerance)
                    new_dataarray.loc[ref_coin_iter.base_assets.values.tolist(),
                                      ref_coin_iter.reference_assets.values.tolist()] = df_to_insert
                    logger.info("The dataframe has been inserted in the new dataframe for"
                                f"{ref_coin_iter.base_assets.values.tolist()}"
                                f"{ref_coin_iter.reference_assets.values.tolist()}.")
                except EmptyDataFrameException:
                    logger.info(f"Empty dataframe for combination of "
                                f"{ref_coin_iter.base_assets.values.tolist()}"
                                f"{ref_coin_iter.reference_assets.values.tolist()}. Skipping the df")
        return new_dataarray

    @staticmethod
    def damage_value_of_dataarray(populated_dataarray):
        coin_name = str(populated_dataarray.base_assets[0].values)
        if coin_name != "ETH":
            coin_index = 0
            while np.equal(populated_dataarray.loc[{"base_assets": coin_name}][0, 0, 0].values, None):
                coin_index += 1
                coin_name = str(populated_dataarray.base_assets[coin_index].values)
                if coin_index>10:
                    raise ValueError

            a = populated_dataarray.loc[{"base_assets": coin_name}]
            a[0, 0, 0] = a[0, 0, 0]+1
            populated_dataarray.loc[{"base_assets": coin_name}] = a
            return populated_dataarray
        else:
            return populated_dataarray

#
# class DataSetDataContainer_OPTIONAL:
#     def __init__(self,
#                  exchange_factory,
#                  base_assets,
#                  reference_assets,
#                  reference_ticker,
#                  aggregate_coordinate_by,
#                  ohlcv_fields,
#                  weight,
#                  start_str,
#                  end_str,
#                  limit):
#         self.aggregate_coordinate_by = aggregate_coordinate_by
#         self.xarray_dataset_container = XArrayDataSetDataContainer(exchange_factory,
#                                                                    base_assets,
#                                                                    reference_assets,
#                                                                    ohlcv_fields,
#                                                                    weight,
#                                                                    start_str,
#                                                                    end_str,
#                                                                    limit)
#         self.reference_base, self.reference_quote = reference_ticker
#         self.xarray_dataset_reference_container = XArrayDataSetDataContainer(exchange_factory,
#                                                                              [self.reference_base],
#                                                                              [self.reference_quote],
#                                                                              ohlcv_fields,
#                                                                              weight,
#                                                                              start_str,
#                                                                              end_str,
#                                                                              limit)
#
#     def general_transfer_from_dataarray_to_dataset(self,
#                                                    dataarray: xr.DataArray,
#                                                    pivot_coordinate: str) -> xr.Dataset:
#         """
#         Transforms a general xr.DataArray to a xr.DataSet
#         Args:
#             dataarray (xr.DataArray): where the ohlcv_fields, base_assets, index_numbers\
#             and references are coordinates of the DataArray
#             pivot_coordinate (str): the coordinate which is translated to the data-vars in xr.DataSet
#
#         Returns:
#             xr.DataSet generated the xr.DataArray and pivoted by pivot_coordinate
#
#         """
#         transposed_dataarray = self.transpose_datarray_over_coord(dataarray, pivot_coordinate)
#         return self.transform_general_dataarray_to_dataset(transposed_dataarray)
#
#     @staticmethod
#     def transpose_datarray_over_coord(dataarray: xr.DataArray,
#                                       pivot_coordinate: str) -> xr.DataArray:
#         """Transposes a general xr.DataArray such that the first-coordinate is the pivot-coordinate
#         Args:
#             dataarray (xr.DataArray): The dataarray whose coordinates are to be transposed
#             pivot_coordinate (str): The coordinate which should the new fundamental coordinate
#         Returns:
#             xr.DataArray whose coordinates are transformed
#         """
#         coords = list(dataarray.coords)
#         assert pivot_coordinate in coords, f"The pivot_coordinate {pivot_coordinate} should be already a coordinate in" \
#                                            f"the dataarray"
#         coords.remove(pivot_coordinate)
#         coords.insert(0, pivot_coordinate)
#         return dataarray.transpose(*coords)
#
#     @staticmethod
#     def transform_general_dataarray_to_dataset(dataarray: xr.DataArray) -> xr.Dataset:
#         """
#         Transforms a general dataarray to a dataset.
#         The fields in the first coordinates is become the data-vars
#         Args:
#             dataarray (xr.DataArray): The dataarray whose data is to be transformed
#         Returns:
#
#         """
#         pivot_coordinate = dataarray.coords.dims[0]
#         return xr.Dataset({individual_dataarray.ohlcv_fields.item():
#                           individual_dataarray.drop(pivot_coordinate)
#                           for individual_dataarray in dataarray})
#
#     async def get_timestamped_data_container(self):
#         return await self.xarray_dataset_container.get_xr_dataset_coin_history()
#

#
#     @staticmethod
#     def get_ascending_list_from_numpy_array(numpy_array):
#         flattened_list = numpy_array.flatten().tolist()
#         filtered_list = list(filter(None, flattened_list))
#         non_duplicated_list = list(set(filtered_list))
#         non_duplicated_list.sort()
#         return non_duplicated_list
#
#     def set_dataset_with_common_coordinate(self,
#                                            dataset: xr.Dataset,
#                                            index_to_replace: str,
#                                            index_of_reference: str,
#                                            reference_ts: List) -> xr.Dataset:
#         assert len(dataset[index_to_replace]) == len(reference_ts), "The length of reference time-stamps do not match" \
#                                                                     f" the dataset['{index_to_replace}']"
#         # dataset["open"] = dataset["open"].fillna(value=np.nan).astype(float)
#         dataset["open_ts"] = dataset.open_ts.fillna(value=-9999).astype(int)
#
#         all_possible_ts = self.get_ascending_list_from_numpy_array(dataset[index_of_reference].values)
#         assinged = dataset.assign_coords(index_number=reference_ts)
#         reindexed_dataset = dataset.reindex({index_to_replace: reference_ts})
#         renamed_dataset = reindexed_dataset.rename({"index_number": "timestamp"})
#         a = 1
#
#
#
#         return dataset
#
#     async def get_dataset_with_common_coordinate(self):
#         original_dataset = await self.get_timestamped_data_container()
#         reference_ts = await self.get_reference_timestamps()
#         return self.set_dataset_with_common_coordinate(original_dataset,
#                                                        "index_number",
#                                                        self.aggregate_coordinate_by,
#                                                        reference_ts)
#

#     async def get_reference_timestamps(self):
#         reference_dataset = await self.get_timestamped_reference_data_container()
#         return self.obtain_coordinate_from_dataset(reference_dataset, self.aggregate_coordinate_by)


class TimeAggregatedDataContainer:
    def __init__(self,
                 exchange_factory,
                 base_assets,
                 reference_assets,
                 ohlcv_fields,
                 start_ts,
                 end_ts,
                 details_of_ts,
                 reference_ticker=("ETH", "BTC"),
                 aggregate_coordinate_by="open_ts"):
        self.exchange_factory = exchange_factory
        self.base_assets = base_assets
        self.reference_assets = reference_assets
        self.ohlcv_fields = ohlcv_fields
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.details_ts = details_of_ts
        self.reference_ticker = reference_ticker
        self.aggregate_coordinate_by = aggregate_coordinate_by

    async def get_time_aggregated_data_container(self):
        interval = "1d"
        time_stamp_indexed_container = TimeStampIndexedDataContainer(
            self.exchange_factory,
            self.base_assets,
            self.reference_assets,
            self.reference_ticker,
            self.aggregate_coordinate_by,
            self.ohlcv_fields,
            interval,
            start_str=self.start_ts,
            end_str=self.end_ts,
            limit=500
        )
        return await time_stamp_indexed_container.get_xr_dataarray_ts_indexed(do_approximation=False)

    def create_chunks_of_requests(self):
        pass
