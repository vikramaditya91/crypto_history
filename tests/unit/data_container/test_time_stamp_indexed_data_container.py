import pytest


class TSIndexedDataContainer:
    pass


@pytest.fixture(scope="session")
def dataarray():
    return TSIndexedDataContainer()


def test_standard_dataarray(dataarray):
    assert 0, dataarray
