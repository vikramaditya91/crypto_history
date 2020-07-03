from __future__ import annotations
from ..utilities import crypto_enum
from itertools import chain
import logging
import re

logger = logging.getLogger(__package__)


class TypedProperty:
    def __init__(self, type_of_property, name):
        self.type_of_property = type_of_property
        self.name = name

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]


class Ticker:
    ticker_name = TypedProperty(str, "ticker_name")
    price = TypedProperty(float, "price")

    possible_reference_coins = {"bitcoin": ["BTC", "XBT"],
                                "ethereum": ["ETH"],
                                "binance": ["BNB"],
                                "altcoins": ["XRP", "TRX"],
                                "currency": ["USDT", "USDC", "BUSD", "TUSD",
                                             "EUR",  "PAX", "USDS",
                                             "TRY", "RUB", "KRW",
                                             "IDRT", "GBP", "UAH",
                                             "IDR", "NGN", "ZAR"]}

    def __init__(self, ticker_name, price=None):
        self.ticker_name = ticker_name
        self.price = price

    def __repr__(self):
        return f"{self.ticker_name}: {self.price}"

    @classmethod
    def create_ticker_from_binance_item(cls, ticker_item):
        return cls(ticker_name=ticker_item["symbol"], price=ticker_item["price"])

    @classmethod
    def get_coin_str_from_ticker(cls, ticker: Ticker) -> str:
        for reference_coin in chain.from_iterable(cls.possible_reference_coins.values()):
            if ticker.ticker_name.endswith(reference_coin) is True:
                return re.sub(f"{reference_coin}(?!.*{reference_coin})", "", ticker.ticker_name)
        logger.debug(f"Ticker {ticker} is ignored as its reference is not recognized")
        return crypto_enum.UNKNOWN_REFERENCE_COIN
