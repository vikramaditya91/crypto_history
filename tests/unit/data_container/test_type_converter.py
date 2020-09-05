import pytest
import numpy as np
from crypto_history import data_container_post
from crypto_history import class_builders
from tests.unit.data_container.sample_data import (  # NOQA
    sample_full_xr_dataarray,
    sample_full_xr_dataset,
    dict_type_of_ohlcv_field,
)


@pytest.fixture
def sample_time_stamp_indexed_data_container():
    """Instance of TimeStampIndexedDataContainer class"""
    binance_factory = class_builders.get("market").get("binance")()
    return data_container_post.TypeConvertedData(binance_factory)


@pytest.mark.asyncio
async def test_setting_type_on_dataarray(
    sample_time_stamp_indexed_data_container,
    sample_full_xr_dataarray,  # NOQA
        dict_type_of_ohlcv_field,  # NOQA
):
    """Test to confirm the types are set correctly on the dataarray"""
    type_set_da = sample_time_stamp_indexed_data_container.\
        set_type_on_dataarray(
            sample_full_xr_dataarray
        )
    for (ohlcv_field, ohlcv_type,) in dict_type_of_ohlcv_field.items():
        np_array = type_set_da.loc[
            {"ohlcv_fields": ohlcv_field}
        ].values.flatten()
        non_nan_np_array = list(
            filter(lambda x: (x == x) and (x is not None), np_array,)
        )
        assert type(np.random.choice(non_nan_np_array)) == ohlcv_type


@pytest.mark.asyncio
async def test_setting_type_on_dataset(
    sample_time_stamp_indexed_data_container,
    sample_full_xr_dataset,  # NOQA
        dict_type_of_ohlcv_field,  # NOQA
):
    """Test to confirm the types are set correctly on the dataset"""
    type_set_ds = sample_time_stamp_indexed_data_container.set_type_on_dataset(
        sample_full_xr_dataset
    )
    for ohlcv_field in type_set_ds:
        np_array = type_set_ds[ohlcv_field].values.flatten()
        non_nan_np_array = list(
            filter(lambda x: (x == x) and (x is not None), np_array,)
        )
        assert (
                type(np.random.choice(non_nan_np_array))
                == dict_type_of_ohlcv_field[ohlcv_field]
        )
