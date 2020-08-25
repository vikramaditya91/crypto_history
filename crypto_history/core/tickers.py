from __future__ import annotations
import logging
from collections import UserList

logger = logging.getLogger(__name__)


class TickerPool(UserList):
    pass


class BinanceTickerPool(TickerPool):
    possible_reference_coins = {
        "bitcoin": ["BTC", "XBT"],
        "ethereum": ["ETH"],
        "binance": ["BNB"],
        "altcoins": ["XRP", "TRX"],
        "currency": [
            "USDT",
            "USDC",
            "BUSD",
            "TUSD",
            "EUR",
            "PAX",
            "USDS",
            "TRY",
            "RUB",
            "KRW",
            "IDRT",
            "GBP",
            "UAH",
            "IDR",
            "NGN",
            "ZAR",
        ],
    }

    futures_key_words = ("BEAR", "BULL", "UP", "DOWN")
