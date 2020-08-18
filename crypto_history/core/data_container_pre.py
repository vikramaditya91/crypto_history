from .primitive_data_builder import PrimitiveDataContainerOperations


class XArrayDataSetDataContainer:
    def __init__(self, *args, **kwargs):
        self.primitive_data_container = PrimitiveDataContainerOperations(*args, **kwargs)

    async def get_primitive_container(self):
        return await self.primitive_data_container.get_populated_primitive_container()

    def transform_datarray_to_dataset(self):
        pass

    def get_xarray_data_set_container(self):
        pass


class TimeStampIndexedDataContainer:
    def __init__(self, *args, **kwargs):
        self.xarray_dataset_container = XArrayDataSetDataContainer(*args, **kwargs)

    async def get_timestamped_data_container(self):
        return await self.xarray_dataset_container.get_primitive_container()


class TimeAggregatedDataContainer:
    def __init__(self,
                 exchange_factory,
                 base_assets,
                 reference_assets,
                 ohlcv_fields,
                 start_ts,
                 end_ts,
                 details_of_ts):
        self.exchange_factory = exchange_factory
        self.base_assets = base_assets
        self.reference_assets = reference_assets
        self.ohlcv_fields = ohlcv_fields
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.details_ts = details_of_ts

    async def get_time_aggregated_data_container(self):
        interval = "1d"
        time_stamp_indexed_container = TimeStampIndexedDataContainer(
            self.exchange_factory,
            self.base_assets,
            self.reference_assets,
            self.ohlcv_fields,
            interval,
            start_str=self.start_ts,
            end_str=self.end_ts,
            limit=500
        )
        return await time_stamp_indexed_container.get_timestamped_data_container()

    def create_chunks_of_requests(self):
        pass
