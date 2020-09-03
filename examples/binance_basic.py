import asyncio
from pprint import pprint
from crypto_history import class_builders, \
    init_logger, data_container_access, data_container_post
import logging
import xarray as xr


async def main():
    init_logger(level=logging.INFO)

    exchange_factory = class_builders.get("market").get("binance")()

    desired_fields = ["open_ts", "open"]
    async with exchange_factory.create_data_homogenizer() as binance_homogenizer:
        await binance_homogenizer.get_all_base_assets()
        base_assets = await binance_homogenizer.get_all_base_assets()
        print(f"All the base assets available on the Binance exchange are {base_assets}")
        time_range = {("25 Jan 2018", "27 Feb 2018"): "1d",
                      ("26 Aug 2020", "now"):         "1w"}
        time_aggregated_data_container = data_container_access.TimeAggregatedDataContainer(
            exchange_factory,
            base_assets=["NANO", "IOST", "XRP"],
            reference_assets=["BTC", "USDT"],
            ohlcv_fields=desired_fields,
            time_range_dict=time_range
        )
        xdataarray_of_coins = await time_aggregated_data_container.get_time_aggregated_data_container()
        pprint(xdataarray_of_coins)

        xdataset = xdataarray_of_coins.to_dataset("ohlcv_fields")
        pprint(xdataset)

        type_converter = data_container_post.TypeConvertedData(exchange_factory)
        type_converted_dataset = type_converter.set_type_on_dataset(xdataset)
        pprint(type_converted_dataset)

        type_converted_dataarray = type_converter.set_type_on_dataarray(xdataarray_of_coins)
        pprint(type_converted_dataarray)



if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # TODO Identify where the unclosed session originates from
