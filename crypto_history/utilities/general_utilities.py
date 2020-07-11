import logging
import asyncio
import re
import functools
from datetime import datetime
from typing import Dict
from abc import ABC


def init_logger(level=logging.INFO):
    log_format = "%(module)s : %(asctime)s : %(levelname)s : %(message)s"
    logging.basicConfig(level=level, format=log_format)


logger = logging.getLogger(__name__)


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
    def __init__(self, request_limit: Dict, pause_seconds=1):
        self.queue = asyncio.Queue()
        self.bucket_list, self.delta_t_list, self.max_requests_list = \
            self.initialize_buckets(request_limit)
        self.last_check = datetime.now()
        self.pause_seconds = pause_seconds
        self.results = []
        self._counter = 0

    @staticmethod
    def initialize_buckets(requests_dict: Dict):
        bucket_list = []
        delta_t_list = []
        max_requests_list = []
        for delta_t, max_requests in requests_dict.items():
            bucket_list.append(max_requests)
            delta_t_list.append(delta_t.total_seconds() * 10e6)
            max_requests_list.append(max_requests)
        return bucket_list, delta_t_list, max_requests_list

    async def hold_if_exceeded(self):
        await self.check_if_within_limits()
        self._counter += 1

    async def check_if_within_limits(self):
        current = datetime.now()
        time_passed = current - self.last_check
        self.last_check = current
        for it in range(len(self.bucket_list)):
            self.bucket_list[it] += time_passed.total_seconds() * 10e6 * self.max_requests_list[it] / self.delta_t_list[
                it]
            if self.bucket_list[it] > self.max_requests_list[it]:
                self.bucket_list[it] = self.max_requests_list[it]
            if self.bucket_list[it] < 1:
                await asyncio.sleep(self.pause_seconds)
                if await self.check_if_within_limits() is True:
                    continue
            self.bucket_list[it] -= 1
        return True


class AbstractFactory(ABC):
    _builders = {}

    @staticmethod
    def valid_subclass_to_register(class_instance):
        if re.match("Concrete.*Factory", class_instance.__name__) is None:
            return False
        return True

    @staticmethod
    def get_identifier_string(class_type):
        matched = re.match("Concrete(.*)Factory", class_type.__name__)
        return matched.group(1).lower()

    @classmethod
    def register_builder(cls, factory_type, class_type):
        assert cls.valid_subclass_to_register(class_type), "Not a valid class to register. Check the name"
        if factory_type not in cls._builders.keys():
            cls._builders[factory_type] = {}
        identifier = cls.get_identifier_string(class_type)
        cls._builders[factory_type][identifier] = class_type

    @classmethod
    def get_builders(cls):
        return cls._builders


def register_factory(factory_type):
    def decorate(decorated_class_type):
        AbstractFactory.register_builder(factory_type, decorated_class_type)

        class Wrapper:
            pass
        Wrapper.__doc__ = decorated_class_type.__doc__
        Wrapper.__name__ = decorated_class_type.__name__
        for attribute, func in decorated_class_type.__dict__.items():
            if callable(decorated_class_type.__dict__[attribute]):
                setattr(Wrapper, attribute, decorated_class_type.__dict__[attribute])
        return Wrapper
    return decorate

