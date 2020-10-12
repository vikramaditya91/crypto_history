import datetime
import re
import math
import pandas as pd


class DateTimeOperations:
    string_match_timedelta_dict = {
        "m": datetime.timedelta(minutes=1),
        "h": datetime.timedelta(hours=1),
        "d": datetime.timedelta(days=1),
        "w": datetime.timedelta(weeks=1),
        # 31 days chosen to not have insufficient lengths
        "M": datetime.timedelta(days=31),
    }

    def map_string_to_timedelta(self, time_string: str) -> datetime.timedelta:
        """
        Maps the string to timedelta
        Args:
            time_string: string which is supposed to represent time

        Returns:
            datetime.timedelta object of the string of time

        """
        if isinstance(time_string, float):
            if math.isnan(time_string):
                return pd.to_timedelta(time_string)
        number_of_items = int(re.search("[0-9]+", time_string).group())
        string_to_match = re.search("[a-zA-Z]+", time_string).group()
        try:
            timedelta_of_string = self.string_match_timedelta_dict[
                string_to_match
            ]
        except KeyError:
            raise KeyError(
                f"{string_to_match} could not match with anything "
                f"in the exchange."
            )
        return datetime.timedelta(
            seconds=(timedelta_of_string.total_seconds() *
                     number_of_items)
        )

    def map_string_to_seconds(self, time_string: str) -> float:
        """
        Maps the string to seconds
        Args:
            time_string: string which is supposed to represent time

        Returns:
            float: seconds of the time_string

        """
        return self.map_string_to_timedelta(time_string).total_seconds()
