from __future__ import annotations
import pathlib
from sqlalchemy.orm import sessionmaker
import xarray as xr
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from sqlalchemy import create_engine
from crypto_history.utilities.general_utilities import register_factory
from crypto_history.utilities.general_utilities import check_for_write_access


class AbstractDiskWriteCreator(ABC):
    """Abstract disk-writer creator"""
    @abstractmethod
    def factory_method(self, *args, **kwargs):
        """factory method to create the disk-writer"""
        pass

    def save_coin_history_to_file(self,
                                  dataarray: xr.DataArray,
                                  file_location: Union[
                                      pathlib.Path, str
                                  ],
                                  ohlcv_to_deposit):
        """
        Save coin history to file
        Args:
            dataarray: xr.DataArray, dataarray which needs to be \
             written to the file
            file_location: path to the filw where it should be written
            ohlcv_to_deposit: list of ohlcv_fields to be deposited

        Returns:

        """
        product = self.factory_method(file_location)
        product.write_coin_history_dataset_to_disk(dataarray,
                                                   ohlcv_to_deposit)


@register_factory(section="write_to_disk", identifier="json")
class JSONCreator(AbstractDiskWriteCreator):
    """JSON creator"""
    def factory_method(self, *args, **kwargs) -> ConcreteAbstractDiskWriter:
        return ConcreteJSONWriter(*args, **kwargs)


@register_factory(section="write_to_disk", identifier="sqlite")
class SQLiteCreator(AbstractDiskWriteCreator):
    """SQLite creator"""
    def factory_method(self, *args, **kwargs) -> ConcreteAbstractDiskWriter:
        return ConcreteSQLiteWriter(*args, **kwargs)


class ConcreteAbstractDiskWriter(ABC):
    """Concrete abstract disk writer"""
    @abstractmethod
    def write_coin_history_dataset_to_disk(self, *args, **kwargs):
        pass


class ConcreteJSONWriter(ConcreteAbstractDiskWriter):
    def __init__(self,
                 json_file_destination):
        self.json_file_destination = json_file_destination

    def write_coin_history_dataset_to_disk(self, *args, **kwargs):
        raise NotImplementedError


class ConcreteSQLiteWriter(ConcreteAbstractDiskWriter):
    """Writes the fields to a SQLite database. It writes a table with
    the name: COIN_HISTORY_{ohlcv}_{reference_asset}_{candle}.

    Warnings: It will raise an error if there are multiple types of candles
        """
    def __init__(self,
                 sqlite_db_path,
                 ):
        self.sqlite_db_path = sqlite_db_path
        self.engine = create_engine(
            f'sqlite:///{self.sqlite_db_path}',
            echo=True
        )

    @staticmethod
    def yield_db_name_from_dataset(dataarray,
                                   ohlcv_to_deposit):
        """
        Yields the df, name of the table from the dataarray
        Args:
            dataarray: xr.DataArray which has to be entered \
             in the SQLite DB
            ohlcv_to_deposit: list of fields to deposit

        Yields: df, name where each df corresponds to one \
            field from the ohlcv fields

        """
        for reference_asset_da in dataarray.reference_assets:
            reference_asset = reference_asset_da.values.tolist()
            for ohlcv in ohlcv_to_deposit:
                to_convert_to_pd = dataarray.loc[:, reference_asset, :, ohlcv]
                df = to_convert_to_pd.transpose().to_pandas()

                non_nan_values = dataarray.loc[
                                 :, reference_asset, :, "weight"
                                 ].to_pandas().dropna()
                unique_values = pd.unique(
                    non_nan_values.values.flatten()
                ).tolist()
                assert len(unique_values) == 1, \
                    f"More than 1 type of weights found. {unique_values}"
                name = f"COIN_HISTORY_{ohlcv}_{reference_asset}_{unique_values[0]}"
                yield df, name

    def write_coin_history_dataset_to_disk(self,
                                           dataarray,
                                           ohlcv_to_deposit):
        """
        Writes the coin history to SQLite DB

        Args:
            dataarray: xr.DataArray which is going to be written in \
            the SQL DB
            ohlcv_to_deposit: types of ohlcv fields to deposit

        """
        if "weight" in ohlcv_to_deposit:
            raise ValueError("weight not expected in ohlcv_to_deposit")

        if check_for_write_access(
                pathlib.Path(self.sqlite_db_path).parent
        ) is True:
            session = sessionmaker(bind=self.engine)()
            for df, name in self.yield_db_name_from_dataset(dataarray,
                                                            ohlcv_to_deposit):
                try:
                    df.to_sql(name, self.engine)
                    session.commit()
                finally:
                    session.close()
        else:
            raise PermissionError(f"Do not have permissions to "
                                  f"write in {self.sqlite_db_path}")


def write_coin_history_to_file(dataarray: xr.DataArray,
                               creator: AbstractDiskWriteCreator,
                               file_location: Union[pathlib.Path, str],
                               *args, **kwargs):
    """
    Writes the coin history to file
    Args:
        dataarray: xr.DataArray, dataset of the coin history
        creator: AbstractDiskWriteCreator, The creator for the \
         disk-writer class
        file_location: location of the file where it should be written
        *args: arguments to be passed to the creator
        **kwargs: keyword-arguments passed to the creator

    """
    if check_for_write_access(pathlib.Path(file_location).parent) is True:
        creator.save_coin_history_to_file(dataarray,
                                          file_location,
                                          *args,
                                          **kwargs)
