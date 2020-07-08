import asyncio
import logging
from abc import ABC, abstractmethod
from binance import client

logger = logging.getLogger(__package__)


class ProducerConsumer:
    def __init__(self, per_second=None, per_minute=None):
        self.queue = asyncio.Queue()
        self.per_second = per_second
        self.per_minute = per_minute

    async def consumer(self):
        pass

    async def producer(self):
        pass

    async def run_tasks(self, *args, **kwargs):
        pass



class RetryModel:
    def __init__(self, retries=3, sleep_seconds=5):
        self._retries = retries
        self.sleep_seconds = sleep_seconds

    async def consume_available_retry(self):
        await asyncio.sleep(self.sleep_seconds)
        return type(self)(self._retries - 1)


class AbstractMarketRequester(ABC):
    def __init__(self):
        self._client = None
        self.retry_strategy_class = RetryModel
        self.request_queue = asyncio.Queue()

    async def request(self, method_name: str, *args, **kwargs):
        retry_strategy_state = self.retry_strategy_class()
        return await self._request_with_retries(method_name,
                                                retry_strategy_state,
                                                *args,
                                                **kwargs)

    @staticmethod
    def _log_request(method_name, *args, **kwargs):
        logger.debug(f"Requesting:\t"
                     f"method: {method_name}\t"
                     f"args: {args}\t"
                     f"kwargs: {''.join(f'{key}: {value}' for key, value in kwargs)}")

    async def _request_with_retries(self, method_name: str, retry_strategy_state=None, *args, **kwargs):
        self._log_request(method_name, *args, **kwargs)
        try:
            return await self._request(method_name, *args, **kwargs)
        except asyncio.exceptions.TimeoutError:
            return await self._retry(method_name, retry_strategy_state, *args, **kwargs)

    async def _retry(self, method_name, retry_strategy_state, *args, **kwargs):
        logger.debug(f"Retrying {method_name} with args:{args}, kwargs:{kwargs}")
        return await self._request_with_retries(method_name,
                                                await retry_strategy_state.consume_available_retry(),
                                                *args,
                                                **kwargs)

    async def _request(self, method_name, *args, **kwargs):
        return await getattr(self._client, method_name)(*args, **kwargs)


class BinanceRequester(AbstractMarketRequester):
    def __init__(self):
        super().__init__()
        self._client = client.AsyncClient(api_key="", api_secret="")


class CoinMarketCapRequester(AbstractMarketRequester):
    pass

