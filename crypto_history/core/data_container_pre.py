from .primitive_data_builder import PrimitiveDataArrayOperations
import xarray as xr
from typing import List

class XArrayDataSetDataContainer:
    """Responsible for transforming the DataArray provided by the primite_data_builder to a more usable DataSet"""
    def __init__(self, *args, **kwargs):
        self.primitive_data_container = PrimitiveDataArrayOperations(*args, **kwargs)

    async def get_primitive_xr_dataarray(self) -> xr.DataArray:
        """
        Gets the primitive xr.DataArray container
        Returns:
            xr.DataArray: the actual xr.DataArray
        """
        return await self.primitive_data_container.get_populated_primitive_container()

    @staticmethod
    def transpose_datarray_over_coord(dataarray: xr.DataArray,
                                      pivot_coordinate: str) -> xr.DataArray:
        """Transposes a general xr.DataArray such that the first-coordinate is the pivot-coordinate
        Args:
            dataarray (xr.DataArray): The dataarray whose coordinates are to be transposed
            pivot_coordinate (str): The coordinate which should the new fundamental coordinate
        Returns:
            xr.DataArray whose coordinates are transformed
        """
        coords = list(dataarray.coords)
        assert pivot_coordinate in coords, f"The pivot_coordinate {pivot_coordinate} should be already a coordinate in" \
                                           f"the dataarray"
        coords.remove(pivot_coordinate)
        coords.insert(0, pivot_coordinate)
        return dataarray.transpose(*coords)

    @staticmethod
    def transform_general_dataarray_to_dataset(dataarray: xr.DataArray) -> xr.Dataset:
        """
        Transforms a general dataarray to a dataset.
        The fields in the first coordinates is become the data-vars
        Args:
            dataarray (xr.DataArray): The dataarray whose data is to be transformed
        Returns:

        """
        pivot_coordinate = dataarray.coords.dims[0]
        return xr.Dataset({individual_dataarray.ohlcv_fields.item():
                          individual_dataarray.drop(pivot_coordinate)
                          for individual_dataarray in dataarray})

    def general_transfer_from_dataarray_to_dataset(self,
                                                   dataarray: xr.DataArray,
                                                   pivot_coordinate: str) -> xr.Dataset:
        """
        Transforms a general xr.DataArray to a xr.DataSet
        Args:
            dataarray (xr.DataArray): where the ohlcv_fields, base_assets, index_numbers\
            and references are coordinates of the DataArray
            pivot_coordinate (str): the coordinate which is translated to the data-vars in xr.DataSet

        Returns:
            xr.DataSet generated the xr.DataArray and pivoted by pivot_coordinate

        """
        transposed_dataarray = self.transpose_datarray_over_coord(dataarray, pivot_coordinate)
        return self.transform_general_dataarray_to_dataset(transposed_dataarray)

    async def get_xr_dataset_coin_history(self):
        """
        Gets the xr.DataSet of the coin histories of the particular chunk.
        Obtains the data and then transforms it to the xr.DataSet
        Returns:
            xr.DataSet data of the coin history
        """
        populated_dataarray = await self.get_primitive_xr_dataarray()
        return self.general_transfer_from_dataarray_to_dataset(populated_dataarray, "ohlcv_fields")


class TimeStampIndexedDataContainer:
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
        self.xarray_dataset_container = XArrayDataSetDataContainer(exchange_factory,
                                                                   base_assets,
                                                                   reference_assets,
                                                                   ohlcv_fields,
                                                                   weight,
                                                                   start_str,
                                                                   end_str,
                                                                   limit)
        self.reference_base, self.reference_quote = reference_ticker
        self.xarray_dataset_reference_container = XArrayDataSetDataContainer(exchange_factory,
                                                                             [self.reference_base],
                                                                             [self.reference_quote],
                                                                             ohlcv_fields,
                                                                             weight,
                                                                             start_str,
                                                                             end_str,
                                                                             limit)

    async def get_timestamped_data_container(self):
        return await self.xarray_dataset_container.get_xr_dataset_coin_history()

    async def get_timestamped_reference_data_container(self):
        return await self.xarray_dataset_reference_container.get_xr_dataset_coin_history()

    @staticmethod
    def set_dataset_with_common_coordinate(dataset: xr.Dataset,
                                           reference_ts: List) -> xr.Dataset:
        a = 1



        return dataset

    async def get_dataset_with_common_coordinate(self):
        original_dataset = await self.get_timestamped_data_container()
        reference_ts = await self.get_reference_timestamps()
        return self.set_dataset_with_common_coordinate(original_dataset, reference_ts)

    @staticmethod
    def obtain_coordinate_from_dataset(dataset: xr.Dataset, coordinate_name: str) -> List:
        coordinate_nested_list = getattr(dataset, coordinate_name).values.tolist()
        return coordinate_nested_list[0][0]

    async def get_reference_timestamps(self):
        reference_dataset = await self.get_timestamped_reference_data_container()
        return self.obtain_coordinate_from_dataset(reference_dataset, self.aggregate_coordinate_by)



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
        return await time_stamp_indexed_container.get_dataset_with_common_coordinate()

    def create_chunks_of_requests(self):
        pass
