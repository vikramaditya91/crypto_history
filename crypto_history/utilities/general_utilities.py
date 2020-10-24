import logging
import asyncio
from datetime import datetime
from typing import Dict, TypeVar
from abc import ABC
from dataclasses import dataclass


logger = logging.getLogger(__name__)


def init_logger(level=logging.INFO):
    log_format = "%(module)s : %(asctime)s : %(levelname)s : %(message)s"
    logging.basicConfig(level=level, format=log_format)


async def gather_dict(tasks: dict):
    async def mark(key, coroutine):
        return key, await coroutine

    return {
        key: result
        for key, result in await asyncio.gather(
            *(mark(key, coroutine) for key, coroutine in tasks.items())
        )
    }


class TokenBucket:
    """Controls the number of requests that can be made to the API.
    All times are written in micro-seconds for a good level of accuracy"""

    def __init__(self, request_limit: Dict, pause_seconds: float = 1):
        """
        Initializes the TokenBucket algorithm to throttle the flow of requests

        Args:
            request_limit: dictionary of requests that can be made where \
            the key is the datetime.timedelta object and the value is the\
             number of maximum requests allowed
            pause_seconds: amount of seconds to pause and test again
        """
        self.queue = asyncio.Queue()
        (
            self.bucket_list,
            self.delta_t_list,
            self.max_requests_list,
        ) = self._initialize_buckets(request_limit)
        self.last_check = datetime.now()
        self.pause_seconds = pause_seconds
        self.results = []
        self._counter = 0

    @staticmethod
    def _initialize_buckets(requests_dict: Dict):
        """
        Initializes the bucket with the tokens.
        Essentially fills up the buckets with various kinds of tokens

        # TODO Does not follow SRP. Split it into different methods
        Args:
            requests_dict(Dict):  the key is the datetime.timedelta object \
            and the value is the number of maximum requests allowed


        Returns:
            Tuple(List): Each item in the dict corresponds to each item\
             in the list
             bucket_list: which is the list of tokens inside the bucket.\
              During initialization it has the max number of tokens which\
               is equal to number of max-requests

            delta_t_list: which is the list of total-duration

            max_requests: which is the list of the max number of requests\
             in allotted time

        """
        bucket_list = []
        delta_t_list = []
        max_requests_list = []
        for delta_t, max_requests in requests_dict.items():
            bucket_list.append(max_requests)
            delta_t_list.append(delta_t.total_seconds() * 10e6)
            max_requests_list.append(max_requests)
        return bucket_list, delta_t_list, max_requests_list

    async def hold_if_exceeded(self):
        """Interface to the outside. It pauses the request \
        until the request is allowed"""
        await self._check_if_within_limits()
        self._counter += 1

    async def _check_if_within_limits(self):
        """Checks if the method can be called i.e if the requests is within the limits
        # TODO Does not follow SRP. Split it into parts"""
        current = datetime.now()
        time_passed = current - self.last_check
        self.last_check = current
        for it in range(len(self.bucket_list)):
            self.bucket_list[it] += (
                time_passed.total_seconds()
                * 10e6
                * self.max_requests_list[it]
                / self.delta_t_list[it]
            )

            if self.bucket_list[it] > self.max_requests_list[it]:
                self.bucket_list[it] = self.max_requests_list[it]
            if self.bucket_list[it] < 1:
                logger.debug(
                    "Requests have exceeded. Waiting for token"
                    " bucket to fill-up"
                )
                await asyncio.sleep(self.pause_seconds)
                if await self._check_if_within_limits() is True:
                    continue
            self.bucket_list[it] -= 1
        return True


class AbstractFactory(ABC):
    """Abstract Factory for all the factories which is responsible
     for registering classes"""

    _builders = {}

    @classmethod
    def register_builder(cls, factory_type, identifier, class_type):
        """Registers the factory to be used later by the user"""
        if factory_type not in cls._builders.keys():
            cls._builders[factory_type] = {}
        cls._builders[factory_type][identifier] = class_type

    @classmethod
    def get_builders(cls):
        return cls._builders


def register_factory(section, identifier):
    """Decorator for registering factories in the factory_types"""
    def decorate(decorated_class_type):
        """Registers the class type in the factory_type"""
        AbstractFactory.register_builder(section,
                                         identifier,
                                         decorated_class_type)

        class Wrapper:
            pass

        Wrapper.__doc__ = decorated_class_type.__doc__
        Wrapper.__name__ = decorated_class_type.__name__
        for attribute, func in decorated_class_type.__dict__.items():
            if callable(decorated_class_type.__dict__[attribute]):
                setattr(
                    Wrapper,
                    attribute,
                    decorated_class_type.__dict__[attribute],
                )
        # Necessary for sphinx to obtain the dc-string correctly
        return Wrapper

    return decorate


class RetryModel:
    """Provides the ability for the method to be retried"""

    def __init__(self, retries: int = 3, sleep_seconds: int = 5):
        """
        Initialize the retry model

        Args:
            retries(int): number of attempts
            sleep_seconds(int): number of seconds to sleep if the\
             retry failed
        """
        self.retries = retries
        self.sleep_seconds = sleep_seconds

    @property
    def retries(self):
        return self._retries

    @retries.setter
    def retries(self, value):
        if value < 1:
            logger.error("All retries have been consumed")
            raise ConnectionError("All retries have been consumed")
        self._retries = value

    async def consume_available_retry(self):
        """
        Triggered after an unsuccessful attempt.

        Returns:
            RetryModel: whose attempts to retry have been reduced by 1

        """
        await asyncio.sleep(self.sleep_seconds)
        return type(self)(retries=self._retries - 1)


def get_dataclass_from_dict(dataclass_name: str, dict_to_convert: Dict):
    """
    Converts a dict to a dataclass of required name
    Args:
        dataclass_name (str): name of the dataclass
        dict_to_convert (Dict): dictionary of items that need to be converted

    Returns:
        dataclass that is generated from the provided dictionary
    """
    dataclass_definition = type(
        dataclass_name,
        (),
        {"__annotations__": {k: type(v) for k, v in dict_to_convert.items()}},
    )
    dataclass_decorated = dataclass(dataclass_definition)
    return dataclass_decorated(**dict_to_convert)


TypeVarPlaceHolder = TypeVar("TypeVarPlaceHolder")
