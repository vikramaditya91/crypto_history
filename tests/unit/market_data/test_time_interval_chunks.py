from crypto_history import class_builders
import pytest
from dateutil import parser



@pytest.fixture
def binance_time_intervals():
    exchange_factory = class_builders.get("market").get("binance")()
    binance_time_intervals = exchange_factory.create_time_interval_chunks()
    return binance_time_intervals


@pytest.mark.parametrize(
    "original_chunk_dict, expected_chunk, exception",
    [
        ({("25 Jan 2020", "27 May 2020"): "1h"},
            [((1579906800000, 1583449200000), '1h'),
             ((1583449200000, 1586988000000), '1h'),
             ((1586988000000, 1590530400000), '1h')],
         None),

        ({("27 May 2020", "27 Aug 2020"): "1d"},
         [((1590530400000, 1598479200000), '1d')],
         None),

        ({("26 Aug 2020", "29 Aug 2020"): "1m"},
         [((1598392800000, 1598444640000), '1m'),
          ((1598444640000, 1598496480000), '1m'),
          ((1598496480000, 1598548320000), '1m'),
          ((1598548320000, 1598600160000), '1m'),
          ((1598600160000, 1598652000000), '1m')]
         , None),

        ({("26 Sep 2020", "now"): "1m"},
         []
         , None),

        ({("26 NOT_A_MONT 2020", "now"): "1m"},
         []
         , parser._parser.ParserError),

        ({("26 Sep 2020", "now"): "1Y"},
         []
         , KeyError),
    ],
)
def test_match_binance_enum(
    original_chunk_dict, expected_chunk, exception, binance_time_intervals
):
    if exception is None:
        resulting_chunk = binance_time_intervals.get_time_range_for_historical_calls(
            original_chunk_dict
        )
        assert resulting_chunk == expected_chunk
    else:
        with pytest.raises(exception):
            binance_time_intervals.get_time_range_for_historical_calls(
                original_chunk_dict
            )
