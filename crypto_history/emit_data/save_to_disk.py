from __future__ import annotations

import datetime
import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Tuple

import pandas as pd
import xarray as xr

from crypto_history.utilities.general_utilities import check_for_write_access, \
    create_dir_if_does_not_exist, register_factory, context_manage_sqlite

logger = logging.getLogger(__package__)


class AbstractDiskWriteCreator(ABC):
    """Abstract disk-writer creator"""

    @abstractmethod
    def factory_method(self, *args, **kwargs):
        """factory method to create the disk-writer"""
        pass

    async def save_coin_history_to_file(self,
                                        data_container_generator,
                                        output_path: Union[
                                            pathlib.Path, str
                                        ],
                                        operations):
        """
        Save coin history to file
        Args:
            data_container_generator: xr.DataArray generator
            output_path: path to the filw where it should be written
            operations: list of operations to perform before depositing

        Returns:

        """
        product = self.factory_method(output_path)
        await product.write_coin_history_dataset_to_disk(data_container_generator,
                                                         operations)


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


@register_factory(section="write_to_disk", identifier="sqlite_pieces")
class SQLitePiecesCreator(AbstractDiskWriteCreator):
    def factory_method(self, *args, **kwargs):
        return ConcreteSQLitePiecesWriter(*args, **kwargs)


class ConcreteAbstractDiskWriter(ABC):
    """Concrete abstract disk writer"""

    @abstractmethod
    def write_coin_history_dataset_to_disk(self, *args, **kwargs):
        pass

    @staticmethod
    def apply_post_operations(da: xr.DataArray,
                              post_ops: List):
        for item in post_ops:
            da = item(da)
        return da


class ConcreteJSONWriter(ConcreteAbstractDiskWriter):
    def __init__(self,
                 json_file_destination):
        self.json_file_destination = json_file_destination

    def write_coin_history_dataset_to_disk(self, *args, **kwargs):
        raise NotImplementedError


class ConcreteSQLiteWriter(ConcreteAbstractDiskWriter):
    """Writes the fields to a SQLite database.

    Warnings: It will raise an error if there are multiple types of candles
    """

    def __init__(self,
                 sqlite_db_path,
                 ):
        self.sqlite_db_path = sqlite_db_path

    @staticmethod
    def get_df_from_da(dataarray: xr.DataArray,
                       reference_asset: str,
                       ohlcv_field: str) -> pd.DataFrame:
        """
        Gets a df from the dataarray whose axis are \
        col-> coins, rows-> timestamps
        Args:
            dataarray: xr.DataArray which will be converted
            reference_asset: reference asset from the dataarray \
             Note that it can have only 1 reference asset
            ohlcv_field: field for flattening

        Returns:
            pd.DataFrame which contains the 2D
            information of the dataarray

        """
        to_convert_to_pd = dataarray.loc[
                           :, reference_asset, :, ohlcv_field
                           ]
        return to_convert_to_pd.transpose().to_pandas()

    @staticmethod
    def get_sql_table_name(dataarray: xr.DataArray,
                           reference_asset: str,
                           ohlcv_field: str
                           ):
        """
        Gets the name of the SQL table. Name is in the format:
        COIN_HISTORY_{ohlcv}_{reference_asset}_{candle}.
        Args:
            dataarray: xr.DataArray which should be converted
            reference_asset: str, reference asset
            ohlcv_field: str, ohlcv_field

        Returns:
            str, a string with the table of the SQL table

        """
        weights = dataarray.loc[
                  :, reference_asset, :, "weight"
                  ].values.flatten()
        unique_values = \
            set(filter(lambda x: isinstance(x, str), weights))
        assert len(unique_values) == 1, \
            f"More than 1 type of weights found. {unique_values}"
        return f"COIN_HISTORY_{ohlcv_field}_" \
               f"{reference_asset}_{unique_values.pop()}"

    def yield_db_name_from_dataset(self,
                                   dataarray: xr.DataArray,
                                   ohlcv_fields: List[str]):
        """
        Yields the df, name of the table from the dataarray
        Args:
            dataarray: xr.DataArray which has to be entered \
             in the SQLite DB
            ohlcv_fields: list of fields to deposit

        Yields: df, name where each df corresponds to one \
            field from the ohlcv fields

        """
        for reference_asset_da in dataarray.reference_assets:
            reference_asset = reference_asset_da.values.tolist()
            for ohlcv_field in ohlcv_fields:
                df = self.get_df_from_da(dataarray,
                                         reference_asset,
                                         ohlcv_field)
                if df.isnull().values.all():
                    logger.warning(f"All the values in the df of {ohlcv_field}"
                                   f" for reference_asset {reference_asset} "
                                   f"are null")
                    continue

                table_name = self.get_sql_table_name(dataarray,
                                                     reference_asset,
                                                     ohlcv_field
                                                     )
                yield df, table_name

    async def write_coin_history_dataset_to_disk(self,
                                                 da_generator,
                                                 operations: Dict[str]):
        """
        Writes the coin history to SQLite DB

        Args:
            da_generator: xr.DataArray to await which is going to be written in \
            the SQL DB
            operations: types of ohlcv fields to deposit

        """
        if "weight" in operations["fields"]:
            raise ValueError("weight not expected in ohlcv_to_deposit")

        if check_for_write_access(
                pathlib.Path(self.sqlite_db_path).parent
        ) is True:
            dataarray = await da_generator.get_time_aggregated_data_container()
            dataarray = self.apply_post_operations(dataarray,
                                                   operations["post"])

            for df, name in self.yield_db_name_from_dataset(dataarray,
                                                            operations["fields"]):
                with context_manage_sqlite(self.sqlite_db_path) as engine:
                    df.to_sql(name, engine)
        else:
            raise PermissionError(f"Do not have permissions to "
                                  f"write in {self.sqlite_db_path}")


class ConcreteSQLitePiecesWriter(ConcreteSQLiteWriter):
    """
    Repeats the obtain coin history chunk, store chunk in DB repeatedly
    """
    def __init__(self,
                 sqlite_dir_path: pathlib.Path):
        self.sqlite_dir_path = sqlite_dir_path
        create_dir_if_does_not_exist(sqlite_dir_path)

    def get_db_path_name(self,
                         time_intervals) -> pathlib.Path:
        """
        Gets the path to the piece of the DB
        Args:
            time_intervals: tuple of start, end times and the type of interval

        Returns:
            path to the pieced DB
        """
        (start_time, end_time), interval = time_intervals
        start = datetime.datetime.fromtimestamp(start_time / 1000)
        end = datetime.datetime.fromtimestamp(end_time / 1000)
        file_name = f'{interval}__' \
                    f'{start.strftime("%d-%m-%Y_%H-%M-%S")}__' \
                    f'{end.strftime("%d-%m-%Y_%H-%M-%S")}.db'
        return pathlib.Path(self.sqlite_dir_path / file_name)

    async def get_chunked_history(self,
                                  da_generator,
                                  post: List,
                                  time_intervals: Tuple):
        """
        Gets the history to be inserted in each DB piece
        Args:
            da_generator: The xr.DataArray generator
            post: post operations needed to be be performed on the da
            time_intervals: tuple of start, end times and the interval type

        Returns:
            da of the chunk of history for that interval

        """
        (start_time, end_time), interval = time_intervals
        chunk_history = await da_generator.get_chunk_history(
            interval, start_time, end_time
        )
        chunk_history = self.apply_post_operations(chunk_history,
                                                   post)
        return chunk_history

    async def write_coin_history_dataset_to_disk(self,
                                                 da_generator,
                                                 operations: Dict[str]):
        """
        Writes the coin history dataset to the disk
        Args:
            da_generator: da generator to write to disk
            operations: dictionary of op

        Returns:

        """
        if "weight" in operations["fields"]:
            raise ValueError("weight not expected in ohlcv_to_deposit")

        if check_for_write_access(
                pathlib.Path(self.sqlite_dir_path)
        ) is True:
            chunks_of_time = da_generator.get_time_interval_chunks(
                da_generator.time_range_dict
            )
            for time_range in chunks_of_time:
                file_path = self.get_db_path_name(time_range)
                if not file_path.exists():
                    chunk_history = await self.get_chunked_history(da_generator,
                                                                   operations["post"],
                                                                   time_range)
                    with context_manage_sqlite(file_path) as engine:
                        for df, name in self.yield_db_name_from_dataset(chunk_history,
                                                                        operations["fields"]):
                            df.to_sql(name, engine)
                else:
                    logger.info(f"{file_path} exists. Skipping time-intervals")
        else:
            raise PermissionError(f"Do not have permissions to "
                                  f"write in {self.sqlite_dir_path}")


async def write_coin_history_to_file(creator: AbstractDiskWriteCreator,
                                     data_container_instance,
                                     output_path: Union[pathlib.Path, str],
                                     *args, **kwargs):
    """
    Writes the coin history to file
    Args:
        data_container_instance: xr.DataArray generator
        creator: AbstractDiskWriteCreator, The creator for the \
         disk-writer class
        output_path: location of the file where it should be written
        *args: arguments to be passed to the creator
        **kwargs: keyword-arguments passed to the creator

    """
    if check_for_write_access(pathlib.Path(output_path).parent) is True:
        await creator.save_coin_history_to_file(data_container_instance,
                                                output_path,
                                                *args,
                                                **kwargs)
