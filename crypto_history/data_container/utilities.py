import pandas as pd
from typing import List, Iterable
from pandas import DataFrame

from crypto_history.utilities import exceptions


class DataFrameOperations:
    """Operations purely dedicated to dataframe"""

    @staticmethod
    def calculate_rows_to_add(
        df: DataFrame, list_of_standard_history: List
    ) -> int:
        """
        Calculates the additional number of rows that might have \
        to be added to get the df in the same shape

        Args:
            df(pd.DataFrame): pandas dataframe which is obtained\
             for the coin's history
            list_of_standard_history: expected standard history\
             which has the complete history

        Returns:
             int: number of rows that have to be added to the df

        """
        df_rows, _ = df.shape
        expected_rows = len(list_of_standard_history)
        return expected_rows - df_rows

    @staticmethod
    def drop_unnecessary_columns_from_df(
        df, necessary_columns: List
    ) -> DataFrame:
        """
        Drop all columns which are not necessary from the df
        Args:
            df (pd.DataFrame): from which the unnecessary columns\
             are to be dropped
            necessary_columns (list): list of columns which are\
             to be stored

        Returns:
            pd.DataFrame where the unnecessary columns are dropped
        """
        unnecessary_columns = [
            col for col in df.columns if col not in necessary_columns
        ]
        return df.drop(unnecessary_columns, axis=1)

    async def get_compatible_df(
        self, standard_example: List, ticker_history: Iterable
    ) -> DataFrame:
        """
        Makes the ticker history compatible to the standard \
        pd.DataFrame by extending the shape to add null values

        Args:
            standard_example (DataFrame): standard example to know \
            how many rows to pad
            ticker_history: history of the current ticker

        Returns:
             pd.DataFrame: compatible history of the df adjusted for \
             same rows as expected

        """
        # TODO Assuming that the df is only not filled in the bottom
        history_df = pd.DataFrame(ticker_history)
        if history_df.empty:
            raise exceptions.EmptyDataFrameException
        padded_df = await self.pad_extra_rows_if_necessary(
            standard_example, history_df
        )
        return padded_df

    @staticmethod
    def add_extra_rows_to_bottom(df: pd.DataFrame, empty_rows_to_add: int):
        """
        Adds extra rows to the pd.DataFrame

        Args:
            df(pd.DataFrame): to which extra rows are to be added
            empty_rows_to_add(int): Number of extra rows that have to be added

        Returns:
             pd.DataFrame: Re-indexed pd.DataFrame which has \
             the compatible number of rows

        """
        new_indices_to_add = list(
            range(df.index[-1] + 1, df.index[-1] + 1 + empty_rows_to_add)
        )
        return df.reindex(df.index.to_list() + new_indices_to_add)

    async def pad_extra_rows_if_necessary(
        self, standard_example: List, history_df: DataFrame
    ) -> DataFrame:
        """
        Add extra rows on the bottom of the DF is necessary. i.e \
        when the history is incomplete.
        # FixMe . Probably needs to be reevaluated
        Args:
            standard_example (DataFrame): standard example on which\
             it is based
            history_df: history of the dataframe that has not been\
             padded yet

        Returns:
            pd.DataFrame: that has been padded

        """
        rows_to_add = self.calculate_rows_to_add(history_df, standard_example)
        if rows_to_add > 0:
            history_df = self.add_extra_rows_to_bottom(history_df, rows_to_add)
        return history_df
