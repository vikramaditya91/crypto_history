import asyncio
import logging
import time
from binance import exceptions
from binance.client import AsyncClient
from abc import ABC
from datetime import timedelta
from crypto_history.utilities.general_utilities import TokenBucket, RetryModel


logger = logging.getLogger(__name__)


class AbstractMarketRequester(ABC):
    """AbstractBaseClass for the low-level market requester"""

    def __init__(self):
        self._client = None
        self.retry_strategy_class = RetryModel
        self.request_queue = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._client.close()

    async def request(self, method_name: str, *args, **kwargs):
        """
        Interface to the user of the low level request made to the API server

        Args:
            method_name: name of the method to be called on the client object.
            *args: arguments transferred by the market operator
            **kwargs: arguments transferred by the market operator

        Returns:
            response from the market/exchange

        """
        retry_strategy_state = self.retry_strategy_class()
        return await self._request_with_retries(
            method_name, retry_strategy_state, *args, **kwargs
        )

    @staticmethod
    def _log_request(method_name, *args, **kwargs):
        logger.debug(
            f"Requesting:\t"
            f"method: {method_name}\t"
            f"args: {args}\t"
            f"kwargs: {''.join(f'{key}: {value}' for key, value in kwargs)}"
        )

    async def _request_with_retries(
        self, method_name: str, retry_strategy_state=None, *args, **kwargs
    ):
        """
        This method is called when the intention is to give it the
         possibility to retry an operations if it times out

        Args:
            method_name: name of the method to be called on the client
            retry_strategy_state: RetryModel class instance
            *args: arguments for the request call
            **kwargs: keyword arguments for the request call

        Returns:
            response from the request

        """
        self._log_request(method_name, *args, **kwargs)
        try:
            return await self._request(method_name, *args, **kwargs)
        except asyncio.exceptions.TimeoutError:
            return await self._retry(
                method_name, retry_strategy_state, *args, **kwargs
            )

    async def _retry(self, method_name, retry_strategy_state, *args, **kwargs):
        """
        Triggered when the original request had failed.

        Args:
            method_name: method for the request
            retry_strategy_state: current state of the RetryModel
            *args: arguments for the request
            **kwargs: keyword arguments for the request

        Returns:
            response from the request

        """
        logger.debug(
            f"Retrying {method_name} with args:{args}, kwargs:{kwargs}"
        )
        return await self._request_with_retries(
            method_name,
            await retry_strategy_state.consume_available_retry(),
            *args,
            **kwargs,
        )

    async def _request(self, method_name, *args, **kwargs):
        """
        The last layer of the request before making the call.
        Holds the request if the queue is close to exceeding the limit
        Args:
            method_name: method for the request
            *args: arguments for the request
            **kwargs: keyword arguments for the request

        Returns:

        """
        await self.request_queue.hold_if_exceeded()
        return await getattr(self._client, method_name)(*args, **kwargs)


class BinanceRequester(AbstractMarketRequester):
    """Low level requester for the Binance API. api_key and \
    api_secret are not required to get market information.\
    Limit for requests are officially set at 2400 per minute. \
    However, it is throttled to 500 per minute
        """

    def __init__(self):
        super().__init__()
        self._client = AsyncClient(api_key="", api_secret="")
        self.request_queue = TokenBucket(
            request_limit={timedelta(minutes=1): 1000}
        )

    async def _request_with_retries(
        self, method_name: str, retry_strategy_state=None, *args, **kwargs
    ):
        """
        Wrapper around the default _request_with_retries to avoid\
         failed requests after too many requests particular to Binance
        Args:
            method_name: name of the method to be called on the client
            retry_strategy_state: RetryModel class instance
            *args: arguments for the request call
            **kwargs: keyword arguments for the request call

        Returns:
            response from the request

        """
        try:
            return await super()._request_with_retries(
                method_name, retry_strategy_state, *args, **kwargs
            )
        except exceptions.BinanceAPIException as e:
            # Error code corresponds to TOO_MANY_REQUESTS in Binance
            if e.code == -1003:
                logger.warning(
                    f"Request could not responds as TOO_MANY_REQUESTS. "
                    f"SYNCHRONOUSLY pausing everything for 30 seconds. "
                    f"Reason {e}"
                )
                time.sleep(30)
                return await self._retry(
                    method_name, retry_strategy_state, *args, **kwargs
                )
            raise


class SomeOtherExchangeRequester(AbstractMarketRequester):
    pass
