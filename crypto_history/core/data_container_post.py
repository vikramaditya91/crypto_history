from abc import ABC, abstractmethod


class AbstractDataContainerIndexManipulator(ABC):
    def __init__(self, data_container):
        self.original_data_container = data_container

    @abstractmethod
    def get_timestamp_indexed_container(self, *args, **kwargs):
        """Sets the index of the data-container as the time-stamp"""
        pass


class XArrayIndexManipulator(AbstractDataContainerIndexManipulator):
    def get_timestamp_indexed_container(self, index_on="open_ts"):

        a = 1
