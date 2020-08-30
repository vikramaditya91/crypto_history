from crypto_history import class_builders
import pytest
import datetime
import pytz
from dateutil import parser


@pytest.fixture
def binance_time_intervals():
    exchange_factory = class_builders.get("market").get("binance")()
    binance_time_intervals = exchange_factory.create_time_interval_chunks()
    return binance_time_intervals


@pytest.mark.parametrize(
    "original_chunk_dict, expected_chunk, exception",
    [
        (
            {("25 Jan 2020 12:00:00 IST", "27 May 2020 12:00:00 IST"): "1h"},
            [
                ((1579933800000, 1583476200000), "1h"),
                ((1583476200000, 1587018600000), "1h"),
                ((1587018600000, 1590561000000), "1h"),
            ],
            None,
        ),
        (
            {("27 May 2020 12:00:00 IST", "27 Aug 2020 12:00:00 IST"): "1d"},
            [((1590561000000, 1598509800000), "1d")],
            None,
        ),
        (
            {("26 Aug 2020 12:00:00 IST", "29 Aug 2020 12:00:00 IST"): "1m"},
            [
                ((1598423400000, 1598475240000), "1m"),
                ((1598475240000, 1598527080000), "1m"),
                ((1598527080000, 1598578920000), "1m"),
                ((1598578920000, 1598630760000), "1m"),
                ((1598630760000, 1598682600000), "1m"),
            ],
            None,
        ),
        ({("26 Sep 2099 12:00:00 IST", "now"): "1m"}, [], None),
        (
            {("26 NOT_A_MONT 2020", "now"): "1m"},
            [],
            parser._parser.ParserError,
        ),
        ({("26 Sep 2020", "now"): "1Y"}, [], KeyError),
    ],
)
def test_match_binance_enum(
    original_chunk_dict, expected_chunk, exception, binance_time_intervals
):
    tzinfos = {"IST": 19800}

    def sanitize_mock_with_timezone(time_string_to_parse):
        if time_string_to_parse == "now":
            return datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        return parser.parse(time_string_to_parse, tzinfos=tzinfos)
    binance_time_intervals.sanitize_item_to_datetime_object = \
        sanitize_mock_with_timezone
    if exception is None:
        resulting_chunk = binance_time_intervals.\
            get_time_range_for_historical_calls(
                original_chunk_dict
            )
        assert resulting_chunk == expected_chunk
    else:
        with pytest.raises(exception):
            binance_time_intervals.get_time_range_for_historical_calls(
                original_chunk_dict
            )
