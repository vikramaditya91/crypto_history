import pytest
import xarray as xr
import numpy as np
import pandas as pd


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
                    np.nan,
                    0.91140000,
                    "0.90170000",
                ],
                [
                    1590451199999,
                    1590537599999,
                    1590623999999,
                    1590710399999,
                    1590796799999,
                ],
                ["1d", "1d", "1d", "1d", 1],
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
def sample_full_xr_dataset(sample_full_xr_dataarray,):
    return sample_full_xr_dataarray.to_dataset("ohlcv_fields")


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
            "weight": ["1d", "1d", "1d", "1d", 1],
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
                [np.nan, np.nan, np.nan],
                [0.91140000, 1590710399999, "1d"],
                ["0.90170000", 1590796799999, 1],
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
                [np.nan, np.nan, np.nan],
                [0.91140000, 1590710399999, "1d"],
                ["0.90170000", 1590796799999, 1],
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


@pytest.fixture
def dict_type_of_ohlcv_field():
    """Dictionary of the expected types of each OHLCV field"""
    return {
        "open": np.float64,
        "open_ts": np.int64,
        "close_ts": np.int64,
        "weight": np.str_,
    }


@pytest.fixture
def xarray_data(request):
    return request.getfuncargvalue(request.param)
