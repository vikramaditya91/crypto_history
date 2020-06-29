from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from binance import client
from ..core.get_market_data import BinanceMarketOperations, CoinMarketCapMarketOperations

logger = logging.getLogger(__package__)


class StockMarketFactory(ABC):
    @abstractmethod
    def create_market_requester(self) -> AbstractMarketRequester:
        pass

    @abstractmethod
    def create_market_operations(self):
        pass

    @abstractmethod
    def create_data_sanitizer(self) -> AbstractDataSanitizer:
        pass


class AbstractMarketRequester(ABC):
    @abstractmethod
    def request(self, *args, **kwargs) -> str:
        pass


class BinanceRequester(AbstractMarketRequester):
    def __init__(self):
        self.client = client.AsyncClient(api_key="", api_secret="")

    async def request(self, binance_func: str, *args, **kwargs):

        logger.debug(f"Obtaining historical klines from Binance for:"
                     f"binance-function: {binance_func}\n"
                     f"args: {args}\t"
                     f"kwargs: {''.join(f'{key}: {value}' for key, value in kwargs)}")
        return await getattr(self.client, binance_func)(*args, **kwargs)


class CoinMarketCapRequester(AbstractMarketRequester):
    def __init__(self):
        raise NotImplementedError

    async def request(self, *args, **kwargs):
        raise NotImplementedError


class ConcreteBinanceFactory(StockMarketFactory):
    def create_market_requester(self) -> BinanceRequester:
        return BinanceRequester()

    def create_market_operations(self, *args, **kwargs) -> BinanceMarketOperations:
        return BinanceMarketOperations(*args, **kwargs)

    def create_data_sanitizer(self) -> BinanceDataSanitizer:
        return BinanceDataSanitizer()


class ConcreteCoinMarketCapFactory(StockMarketFactory):
    def create_market_requester(self) -> CoinMarketCapRequester:
        return CoinMarketCapRequester()

    def create_market_operations(self, *args, **kwargs) -> CoinMarketCapMarketOperations:
        return CoinMarketCapMarketOperations(*args, **kwargs)

    def create_data_sanitizer(self) -> CoinMarketCapDataSanitizer:
        return CoinMarketCapDataSanitizer()


class AbstractDataSanitizer(ABC):
    @abstractmethod
    def clean_and_manipulate_data(self, *args, **kwargs) -> None:
        pass


class BinanceDataSanitizer(AbstractDataSanitizer):
    def clean_and_manipulate_data(self, historical_data):
        print(f"Historical data is being cleaned on A")


class CoinMarketCapDataSanitizer(AbstractDataSanitizer):
    def clean_and_manipulate_data(self, historical_data):
        print(f"Historical data is being cleaned on B")







