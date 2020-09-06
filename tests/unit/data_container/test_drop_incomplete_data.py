import pytest
from crypto_history import data_container_post
from tests.unit.data_container.sample_data import (  # NOQA
    sample_full_xr_dataarray,
    sample_full_xr_dataset,
    dict_type_of_ohlcv_field,
    xarray_data
)


@pytest.fixture
def sample_incomplete_data_dropper():
    """Instance of HandleIncompleteData class"""
    return data_container_post.HandleIncompleteData(
        coordinates_to_drop=["base_assets", "reference_assets"]
    )


@pytest.mark.parametrize('xarray_data',
                         ['sample_full_xr_dataarray',
                          'sample_full_xr_dataset'])
def test_drop_dataarray_coins_with_entire_na(
    sample_incomplete_data_dropper,
    xarray_data, # NOQA
    request
):
    """Test to confirm that the coins are dropped if
    dataarray/dataset is completely empty"""
    xarray_data = request.getfixturevalue(xarray_data)
    assert list(xarray_data.reference_assets.values) \
           == ["USDT", "XRP"]
    data_dropped_da = \
        sample_incomplete_data_dropper.\
        drop_xarray_coins_with_entire_na(
            xarray_data
        )
    assert list(data_dropped_da.reference_assets.values) == ["USDT"]


def test_dataarray_nullified_if_any_entry_is_nan(
    sample_incomplete_data_dropper,
    sample_full_xr_dataarray, # NOQA
):
    """Test to confirm that the dataarray
    is nullified if even 1 entry is nan"""
    sample_coord_removed_dataarray = sample_incomplete_data_dropper.\
        drop_xarray_coins_with_entire_na(sample_full_xr_dataarray)

    assert False is sample_coord_removed_dataarray.\
        loc[{"base_assets": "NANO"}].isnull().all().item()

    nullified_dataarray = sample_incomplete_data_dropper.\
        nullify_incomplete_data_from_dataarray(
            sample_coord_removed_dataarray
        )

    assert True is nullified_dataarray.\
        loc[{"base_assets": "NANO"}].isnull().all().item()


def test_dataset_nullified_if_any_entry_is_nan(
    sample_incomplete_data_dropper,
    sample_full_xr_dataset, # NOQA
):
    """Test to confirm that the dataset
    is nullified if even 1 entry is nan"""
    sample_coord_removed_dataset = sample_incomplete_data_dropper.\
        drop_xarray_coins_with_entire_na(sample_full_xr_dataset)

    assert False is sample_coord_removed_dataset.\
        sel(base_assets="NANO").isnull().all().\
        to_array().any().item()

    nullified_dataarray = sample_incomplete_data_dropper.\
        nullify_incomplete_data_from_dataset(sample_coord_removed_dataset)

    assert True is nullified_dataarray.\
        sel(base_assets="NANO").isnull().all().\
        to_array().any().item()
