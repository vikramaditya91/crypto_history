import asyncio
from pprint import pprint
from crypto_history import class_builders


async def main():
    exchange_factory = class_builders.get("market").get("binance")()
    data_container_factory = class_builders.get("data").get("xarray")()

    coin_history_obtainer = await data_container_factory.create_coin_history_obtainer(exchange_factory,
                                                                                      interval="1d",
                                                                                      start_str="1 January 2020",
                                                                                      end_str="4 June 2020",
                                                                                      limit=1000
                                                                                      )
    data_operations = await data_container_factory.create_data_container_operations(coin_history_obtainer)
    data_container = await data_operations.get_filled_container()
    pprint(data_container)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
