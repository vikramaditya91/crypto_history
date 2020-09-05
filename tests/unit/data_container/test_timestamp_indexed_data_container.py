import pytest
from mock import Mock
import xarray as xr
from crypto_history import data_container_access
from crypto_history import class_builders
from ...helpers_test_utilities import async_return
from tests.unit.data_container.sample_data import (  # NOQA
    sample_full_xr_dataarray,
    sample_approximated_time_indexed_array,
    sample_exact_time_indexed_array,
    sample_reference_xr_datarray,
)


@pytest.fixture
async def sample_time_stamp_indexed_data_container():
    """Instance of TimeStampIndexedDataContainer class"""
    binance_factory = class_builders.get("market").get("binance")()
    async with data_container_access.\
            TimeStampIndexedDataContainer.\
            create_time_stamp_indexed_data_container(
                exchange_factory=binance_factory,
                base_assets=["NANO", "XMR"],
                reference_assets=["USDT", "XRP"],
                reference_ticker=("ETH", "BTC",),
                aggregate_coordinate_by="open_ts",
                ohlcv_fields=["open_ts", "open", "close_ts"],
                weight="1d",
                start_time="25 May 2020",
                end_time="29 May 2020",
            ) as timestamp_indexed_container:
        yield timestamp_indexed_container


@pytest.mark.asyncio
async def test_get_primitive_time_approx_xr_dataarray(
    sample_time_stamp_indexed_data_container, # NOQA
    sample_full_xr_dataarray, # NOQA
    sample_reference_xr_datarray, # NOQA
    sample_approximated_time_indexed_array, # NOQA
):
    """Test to confirm the approximated time-stamp indexes"""
    sample_time_stamp_indexed_data_container.\
        get_primitive_full_xr_dataarray = Mock(
            return_value=async_return(sample_full_xr_dataarray)
        )
    sample_time_stamp_indexed_data_container.\
        get_primitive_reference_xr_dataarray = Mock(
            return_value=async_return(sample_reference_xr_datarray)
        )
    sample_time_stamp_indexed_data_container.aggregate_coordinate_by = (
        "open_ts"
    )
    time_indexed_dataarray = await \
        sample_time_stamp_indexed_data_container.\
        get_xr_dataarray_indexed_by_timestamps(
            do_approximation=True, tolerance_ratio=0.001,
        )
    expected_dataarray = xr.DataArray(
        sample_approximated_time_indexed_array,
        dims=("base_assets", "reference_assets", "timestamp", "ohlcv_fields",),
        coords={
            "base_assets": ["NANO", "XMR"],
            "reference_assets": ["USDT", "XRP"],
            "timestamp": [
                1590364800000,
                1590451200000,
                1590537600000,
                1590624000000,
                1590710400000,
            ],
            "ohlcv_fields": ["open", "close_ts", "weight"],
        },
    )
    xr.testing.assert_identical(
        expected_dataarray, time_indexed_dataarray,
    )


@pytest.mark.asyncio
async def test_get_primitive_exact_xr_dataarray(
    sample_time_stamp_indexed_data_container, # NOQA
    sample_full_xr_dataarray, # NOQA
    sample_reference_xr_datarray, # NOQA
    sample_exact_time_indexed_array, # NOQA
):
    """Test to confirm the exact time-stamp indexes"""
    sample_time_stamp_indexed_data_container.\
        get_primitive_full_xr_dataarray = Mock(
            return_value=async_return(sample_full_xr_dataarray)
        )
    sample_time_stamp_indexed_data_container.\
        get_primitive_reference_xr_dataarray = Mock(
            return_value=async_return(sample_reference_xr_datarray)
        )
    sample_time_stamp_indexed_data_container.aggregate_coordinate_by = (
        "open_ts"
    )
    time_indexed_dataarray = await \
        sample_time_stamp_indexed_data_container.\
        get_xr_dataarray_indexed_by_timestamps(
            do_approximation=False, tolerance_ratio=0.001,
        )
    expected_dataarray = xr.DataArray(
        sample_exact_time_indexed_array,
        dims=("base_assets", "reference_assets", "timestamp", "ohlcv_fields",),
        coords={
            "base_assets": ["NANO", "XMR"],
            "reference_assets": ["USDT", "XRP"],
            "timestamp": [
                1590364800000,
                1590364800001,
                1590451200000,
                1590537600000,
                1590624000000,
                1590710400000,
            ],
            "ohlcv_fields": ["open", "close_ts", "weight"],
        },
    )
    xr.testing.assert_identical(
        expected_dataarray, time_indexed_dataarray,
    )
