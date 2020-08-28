import asyncio


def async_return(result):
    """Provides a future which can be mocked to be the result
     value of an awaited function"""
    future = asyncio.Future()
    future.set_result(result)
    return future
