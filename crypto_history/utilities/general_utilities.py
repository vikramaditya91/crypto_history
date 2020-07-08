import logging
import asyncio


def init_logger(level=logging.INFO):
    logging.basicConfig(level=level)


async def gather_dict(tasks: dict):
    async def mark(key, coroutine):
        return key, await coroutine

    return {
        key: result
        for key, result in await asyncio.gather(
            *(mark(key, coroutine) for key, coroutine in tasks.items())
        )
    }
