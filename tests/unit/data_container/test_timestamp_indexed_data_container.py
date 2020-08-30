import pytest
import numpy as np
import pandas as pd
from mock import Mock
import xarray as xr
from crypto_history import data_container_intra
from crypto_history import class_builders
from ...helpers_test_utilities import async_return


@pytest.fixture
def sample_time_stamp_indexed_data_container():
    """Instance of TimeStampIndexedDataContainer class"""
    binance_factory = class_builders.get("market").get("binance")()
    timestamp_indexed_container = \
        data_container_intra.TimeStampIndexedDataContainer(
            exchange_factory=binance_factory,
            base_assets=["NANO", "XMR"],
            reference_assets=["USDT", "XRP"],
            reference_ticker=("ETH", "BTC"),
            aggregate_coordinate_by="open_ts",
            ohlcv_fields=["open_ts", "open", "close_ts"],
            weight="1d",
            start_time="25 May 2020",
            end_time="29 May 2020",
        )
    return timestamp_indexed_container


@pytest.fixture
def sample_full_xr_dataarray():
    """Full primitive (non time-indexed) xr.DataArray"""
    full_array = [
        [
            [
                [
                    1590364800000,
                    1590451200000,
                    1590537600000,
                    1590624000000,
                    1590710400000,
                ],
                [
                    "0.88150000",
                    "0.93990000",
                    "0.89100000",
                    "0.91140000",
                    "0.90170000",
                ],
                [
                    1590451199999,
                    1590537599999,
                    1590623999999,
                    1590710399999,
                    1590796799999,
                ],
                ["1d", "1d", "1d", "1d", "1d"],
            ],
            [
                [None, None, None, None, None],
                [None, None, None, None, None],
                [None, None, None, None, None],
                [None, None, None, None, None],
            ],
        ],
        [
            [
                [
                    1590364800001,
                    1590451200000,
                    1590537600000,
                    1590624000000,
                    1590710400000,
                ],
                [
                    "60.65000000",
                    "61.58000000",
                    "61.91000000",
                    "64.50000000",
                    "67.13000000",
                ],
                [
                    1590451199999,
                    1590537599999,
                    1590623999999,
                    1590710399999,
                    1590796799999,
                ],
                ["1d", "1d", "1d", "1d", "1d"],
            ],
            [
                [None, None, None, None, None],
                [None, None, None, None, None],
                [None, None, None, None, None],
                [None, None, None, None, None],
            ],
        ],
    ]
    full_dataarray = xr.DataArray(
        full_array,
        dims=(
            "base_assets",
            "reference_assets",
            "ohlcv_fields",
            "index_number",
        ),
        coords={
            "base_assets": ["NANO", "XMR"],
            "reference_assets": ["USDT", "XRP"],
            "ohlcv_fields": ["open_ts", "open", "close_ts", "weight"],
            "index_number": [0, 1, 2, 3, 4],
        },
    )
    return full_dataarray


@pytest.fixture
def sample_reference_xr_datarray():
    """Reference primitive (non time-indexed) xr.DataArray"""
    reference_df = pd.DataFrame(
        {
            "open_ts": [
                1590364800000,
                1590451200000,
                1590537600000,
                1590624000000,
                1590710400000,
            ],
            "open": [
                "0.02292400",
                "0.02293700",
                "0.02272600",
                "0.02263400",
                "0.02300000",
            ],
            "close_ts": [
                1590451199999,
                1590537599999,
                1590623999999,
                1590710399999,
                1590796799999,
            ],
            "weight": ["1d", "1d", "1d", "1d", "1d"],
        }
    )
    reference_dataarray = xr.DataArray(
        None,
        dims=(
            "base_assets",
            "reference_assets",
            "ohlcv_fields",
            "index_number",
        ),
        coords={
            "base_assets": ["BTC"],
            "reference_assets": ["ETH"],
            "ohlcv_fields": ["open_ts", "open", "close_ts", "weight"],
            "index_number": [0, 1, 2, 3, 4],
        },
    )
    reference_dataarray.loc["BTC", "ETH"] = reference_df.transpose()
    return reference_dataarray


@pytest.fixture
def sample_approximated_time_indexed_array():
    """Array of values of approximated time-indexed xr.DataArray"""
    return [
        [
            [
                ["0.88150000", 1590451199999, "1d"],
                ["0.93990000", 1590537599999, "1d"],
                ["0.89100000", 1590623999999, "1d"],
                ["0.91140000", 1590710399999, "1d"],
                ["0.90170000", 1590796799999, "1d"],
            ],
            [
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
            ],
        ],
        [
            [
                ["60.65000000", 1590451199999, "1d"],
                ["61.58000000", 1590537599999, "1d"],
                ["61.91000000", 1590623999999, "1d"],
                ["64.50000000", 1590710399999, "1d"],
                ["67.13000000", 1590796799999, "1d"],
            ],
            [
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
            ],
        ],
    ]


@pytest.fixture
def sample_exact_time_indexed_array():
    """Array of values of exact time-indexed xr.DataArray"""
    return [
        [
            [
                ["0.88150000", 1590451199999, "1d"],
                [np.nan, np.nan, np.nan],
                ["0.93990000", 1590537599999, "1d"],
                ["0.89100000", 1590623999999, "1d"],
                ["0.91140000", 1590710399999, "1d"],
                ["0.90170000", 1590796799999, "1d"],
            ],
            [
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
            ],
        ],
        [
            [
                [np.nan, np.nan, np.nan],
                ["60.65000000", 1590451199999, "1d"],
                ["61.58000000", 1590537599999, "1d"],
                ["61.91000000", 1590623999999, "1d"],
                ["64.50000000", 1590710399999, "1d"],
                ["67.13000000", 1590796799999, "1d"],
            ],
            [
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
            ],
        ],
    ]


@pytest.mark.asyncio
async def test_get_primitive_time_approx_xr_dataarray(
    sample_time_stamp_indexed_data_container,
    sample_full_xr_dataarray,
    sample_reference_xr_datarray,
    sample_approximated_time_indexed_array,
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
    time_indexed_dataarray = await sample_time_stamp_indexed_data_container.\
        get_xr_dataarray_indexed_by_timestamps(
            do_approximation=True, tolerance_ratio=0.001
        )
    expected_dataarray = xr.DataArray(
        sample_approximated_time_indexed_array,
        dims=("base_assets", "reference_assets", "timestamp", "ohlcv_fields"),
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
    xr.testing.assert_identical(expected_dataarray, time_indexed_dataarray)


@pytest.mark.asyncio
async def test_get_primitive_exact_xr_dataarray(
    sample_time_stamp_indexed_data_container,
    sample_full_xr_dataarray,
    sample_reference_xr_datarray,
    sample_exact_time_indexed_array,
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
    time_indexed_dataarray = await sample_time_stamp_indexed_data_container.\
        get_xr_dataarray_indexed_by_timestamps(
            do_approximation=False, tolerance_ratio=0.001
        )
    expected_dataarray = xr.DataArray(
        sample_exact_time_indexed_array,
        dims=("base_assets", "reference_assets", "timestamp", "ohlcv_fields"),
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
    xr.testing.assert_identical(expected_dataarray, time_indexed_dataarray)
