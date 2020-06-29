from __future__ import annotations
import pandas as pd
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod


class DataContainerCreator:
    @abstractmethod
    def factory_method(self):
        pass

    async def weed_out_the_undesirables(self):
        data_container = self.factory_method()


class XArrayCreator(DataContainerCreator):
    def factory_method(self):
        return XArrayContainer()


class SQLConnectCreator(DataContainerCreator):
    def factory_method(self):
        return SQLConnectContainer()


class DataContainer(ABC):
    @abstractmethod
    def populate_container(self):
        return


class XArrayContainer(DataContainer):
    def __init__(self):
        self.container = xr.DataArray()

    def populate_container(self):
        return


class SQLConnectContainer(DataContainer):
    def populate_container(self):
        raise NotImplementedError
