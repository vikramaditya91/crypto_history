from __future__ import annotations
from ..utilities import crypto_enum
from binance.client import AsyncClient
from collections import namedtuple
from itertools import chain
from functools import lru_cache
from abc import ABC, abstractmethod
import logging
from collections import UserList
from dataclasses import dataclass
from itertools import groupby
import re

logger = logging.getLogger(__name__)


class TickerPool(UserList):
    possible_reference_coins = {"bitcoin": ["BTC", "XBT"],
                                "ethereum": ["ETH"],
                                "binance": ["BNB"],
                                "altcoins": ["XRP", "TRX"],
                                "currency": ["USDT", "USDC", "BUSD", "TUSD",
                                             "EUR", "PAX", "USDS",
                                             "TRY", "RUB", "KRW",
                                             "IDRT", "GBP", "UAH",
                                             "IDR", "NGN", "ZAR"]}

    futures_key_words = ("BEAR", "BULL", "UP", "DOWN")

    @abstractmethod
    def obtain_unique_items(self, *args, **kwargs):
        pass


class BinanceTickerPool(TickerPool):
    possible_reference_coins = {"bitcoin": ["BTC", "XBT"],
                                "ethereum": ["ETH"],
                                "binance": ["BNB"],
                                "altcoins": ["XRP", "TRX"],
                                "currency": ["USDT", "USDC", "BUSD", "TUSD",
                                             "EUR",  "PAX", "USDS",
                                             "TRY", "RUB", "KRW",
                                             "IDRT", "GBP", "UAH",
                                             "IDR", "NGN", "ZAR"]}

    futures_key_words = ("BEAR", "BULL", "UP", "DOWN")

    def obtain_unique_items(self, attribute_to_isolate: str):
        grouped_tickers = groupby(self, key=lambda x: getattr(x, attribute_to_isolate))
        unique_items = []
        for unique_item, _ in grouped_tickers:
            unique_items.append(unique_item)
        return unique_items

