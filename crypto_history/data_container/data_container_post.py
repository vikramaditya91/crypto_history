import logging
import itertools
import xarray as xr
import pandas as pd
import numpy as np
from typing import Union, Dict
from collections import OrderedDict
from crypto_history.stock_market.stock_market_factory import StockMarketFactory
from crypto_history.utilities import datetime_operations

logger = logging.getLogger(__package__)


class TypeConvertedData:
    """Type converts the data in the dataarray/dataset"""

    type_mapping = {
        int: pd.Int64Dtype,
        str: pd.StringDtype,
        float: np.float64,
    }

    def __init__(self,
                 exchange_factory: StockMarketFactory):
        """Initializes the type conversion of the data"""
        self.ohlcv_field_types = exchange_factory.create_ohlcv_field_types()

    def get_ohlcv_field_type_dict(self) -> Dict:
        """
        Gets the field types of the OHLCV Fields converted to \
        the numpy/pandas format. This is done to be able to \
        handle nan values in Int/String types

        Returns (dict): Dictionary of the map from the \
        ohlcv-field to the pd/np type

        """
        original_ohlcv_field_type = self.ohlcv_field_types.get_dict_name_type()
        original_ohlcv_field_type["weight"] = str
        return dict(
            map(
                lambda x: (x[0], self.type_mapping[x[1]]),
                original_ohlcv_field_type.items(),
            )
        )

    def set_type_on_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Sets the type on the xr.DataSet according to the \
        ohlcv field type

        Args:
            dataset(xr.DataSet): The dataset on which the type \
                has to be set

        Returns:
            xr.DataSet which has the type set on it
        """
        keys_in_dataset = list(dataset.keys())
        dict_of_types = self.get_ohlcv_field_type_dict()
        for key in keys_in_dataset:
            dataset[key] = dataset[key].astype(dict_of_types[key])
        return dataset

    def set_type_on_dataarray(self, dataarray: xr.DataArray) -> xr.DataArray:
        """
        Sets the type on the xr.DataArray according to the \
            ohlcv field type
        Args:
            dataarray(xr.DataArray): The DataArray on which the type \
                has to be set

        Returns:
            xr.DataArray which has the type set on it
        """
        dict_of_types = self.get_ohlcv_field_type_dict()
        for ohlcv_field in dataarray.ohlcv_fields.values:
            dataarray.loc[{"ohlcv_fields": ohlcv_field}] = dataarray.loc[
                {"ohlcv_fields": ohlcv_field}
            ].astype(dict_of_types[ohlcv_field])
        return dataarray


class HandleIncompleteData:
    """Responsible for handling missing data:\
    1. If a certain coin has to be dropped if it is null
    2. If a ticker has to be nulliifed as it has incomplete data
    """
    def __init__(self, coordinates_to_drop=None):
        """Initializes the incomplete data. The iterations on the
         coordinates are set with coordinates to drop"""
        if coordinates_to_drop is None:
            coordinates_to_drop = ["base_assets", "reference_assets"]
        self.coordinates_to_drop = coordinates_to_drop

    def drop_xarray_coins_with_entire_na(
        self, data_item: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Drops the coins from the base/reference asset if all \
            its corresponding values are nan
        Args:
            data_item(xr.DataArray/xr.DataSet): which contains \
                information of the coin histories

        Returns:
            xr.DataArray/xr.DataSet where the coins have been dropped

        """
        logger.debug(f"{type(data_item)} prior to dropping entire "
                     f"data has coordinates: {data_item.coords}")
        for coordinate in self.coordinates_to_drop:
            data_item = data_item.dropna(coordinate, how="all")
        logger.debug(f"{type(data_item)} after dropping entire "
                     f"data has coordinates: {data_item.coords}")
        return data_item

    def get_all_coord_combinations(self,
                                   data_item: Union[xr.DataArray, xr.Dataset]):
        """
        Gets all the various combinations to iterate
         according to the coordinates to drop

        Args:
            data_item (xr.DataArray/xr.DataSet): data_item whose combinations
             need to be iterated over

        Yields:
            A dict with various combinations

        """
        combinations = OrderedDict()
        for coordinate in self.coordinates_to_drop:
            combinations[coordinate] = list(
                getattr(data_item, coordinate).values
            )
        keys, values = zip(*combinations.items())

        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def nullify_incomplete_data_from_dataarray(
        self, dataarray: xr.DataArray
    ) -> xr.DataArray:
        """
        Nullifies incomplete data from the xr.DataArray
        Args:
            dataarray (xr.DataArray): dataarray whose \
                coordinates are to be nullified

        Returns:
            xr.DataArray whose data has been nullified if incomplete

        """

        for combination in self.get_all_coord_combinations(dataarray):
            if dataarray.loc[combination].isnull().any():
                dataarray.loc[combination] = np.nan
        return dataarray

    def nullify_incomplete_data_from_dataset(
        self, dataset: xr.Dataset
    ) -> xr.Dataset:
        """
        Nullifies the incomplete data of datasets

        Notes:
            Using indexing to assign values to a
             subset of dataset (e.g., ds[dict(space=0)] = 1) is\
              not yet supported.
             http://xarray.pydata.org/en/stable/indexing.html
        Args:
            dataset(xr.DataSet): dataset whose data is to be nullified

        Returns:
            xr.DataSet whose incomplete items are nullified

        """
        for combination in self.get_all_coord_combinations(dataset):
            if any(list(dataset.loc[combination].isnull().any().values())):
                for ohlcv_field in list(dataset.keys()):
                    dataset[ohlcv_field].loc[combination] = np.nan
        return dataset


class ApplyWeightToDataArray:
    @staticmethod
    def get_weights_numpy_float_array(numpy_array: np.ndarray) -> np.ndarray:
        """
        Obtains the numpy array in float values from the string
        Args:
            numpy_array: np.ndarray numpy array of strings

        Returns: np.ndarray numpy array of seconds (float values)

        """
        date_time_operator = datetime_operations.DateTimeOperations()
        string_to_seconds = date_time_operator.map_string_to_seconds
        return np.vectorize(string_to_seconds)(numpy_array)

    @staticmethod
    def correct_shaped_np_array(numpy_array: np.ndarray,
                                len_ohlcv_fields: int) -> np.ndarray:
        """
        Fix the shape of the np array by filling in the shape
        Args:
            numpy_array: np.ndarray whose shape is not ok
            len_ohlcv_fields: int of the duplication required

        Returns: np.ndarray whose shape is as expected for the da

        """
        # TODO This should ideally do setting indices programatically
        numpy_array_dim_added = np.expand_dims(numpy_array, 3)
        return np.tile(numpy_array_dim_added, (1, 1, 1, len_ohlcv_fields))

    def get_weights_dataarray(self,
                              original_dataarray: xr.DataArray
                              ) -> xr.DataArray:
        """
        Gets the da of weights
        Args:
            original_dataarray: xr.DataArray which is not weighted,\
             but has a "weight" data_var

        Returns: xr.DataArray which is weighted

        """
        numpy_weights = self.get_weights_numpy_float_array(
            original_dataarray.loc[:, :, :, "weight"]
        )
        well_shaped_np_array = self.correct_shaped_np_array(
            numpy_weights,
            len(original_dataarray.ohlcv_fields)
        )
        return xr.DataArray(
            well_shaped_np_array,
            dims=original_dataarray.dims,
            coords=original_dataarray.coords
        )

    def get_weighted_data_container(self,
                                    original_dataarray: xr.DataArray):
        """
        Gets the weighted xr.DataArray
        Args:
            original_dataarray: xr.DataArray which has to be weighted

        Returns: xr.DataArray which is weighted

        """
        weights_dataarray = self.get_weights_dataarray(original_dataarray)
        return original_dataarray.weighted(weights_dataarray)

    def switch_str_to_float(self,
                            original_da: xr.DataArray
                            ) -> xr.DataArray:
        """
        Switches a dataarray of strings to dataarray of floats
        Args:
            original_da: xr.DataArray whose values are str

        Returns: xr.DataArray whose values are float

        """
        float_np = self.get_weights_numpy_float_array(original_da.values)
        return xr.DataArray(
            float_np,
            dims=original_da.dims,
            coords=original_da.coords
        )
