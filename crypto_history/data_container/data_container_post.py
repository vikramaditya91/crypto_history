import xarray as xr
import contextlib
import numpy as np
from ..stock_market import stock_market_factory


class TypeConvertedData:
    def __init__(self, exchange_factory):
        self.ohlcv_field_types = exchange_factory.create_ohlcv_field_types()

    @staticmethod
    def get_keys_of_dataset(dataset: xr.Dataset):
        return list(dataset.keys())

    def get_ohlcv_field_type_dict(self):
        ohlcv_fields = self.ohlcv_field_types.get_dict_name_type()
        ohlcv_fields["weight"] = str
        return ohlcv_fields

    def set_type_on_dataset(self,
                            dataset: xr.Dataset):
        keys_in_dataset = self.get_keys_of_dataset(dataset)
        dict_of_types = self.get_ohlcv_field_type_dict()
        for key in keys_in_dataset:
            dataset[key] = dataset[key].astype(dict_of_types[key])
        return dataset

    def set_type_on_dataarray(self,
                              dataarray: xr.DataArray):
        dict_of_types = self.get_ohlcv_field_type_dict()
        for ohlcv_field in dataarray.ohlcv_fields.values:
            dataarray.loc[{"ohlcv_fields": ohlcv_field}] = \
                dataarray.loc[{"ohlcv_fields": ohlcv_field}].\
                astype(dict_of_types[ohlcv_field])
        return dataarray

class HandleIncompleteData:
    def __init__(self):
        pass

    # def drop_assets_with_incomplete_data