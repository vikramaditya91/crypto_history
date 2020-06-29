class TypedProperty:
    def __init__(self, type_of_property):
        self.type_of_property = type_of_property

    def __set__(self, instance, value):
        self.value = self.type_of_property(value)

    def __get__(self, instance, owner):
        return self.value


class Ticker:
    ticker_name = TypedProperty(str)
    price = TypedProperty(float)

    def __init__(self, ticker_name, price=None):
        self.ticker_name = ticker_name
        self.price = price

    def __repr__(self):
        return f"{self.ticker_name}: {self.price}"

    @classmethod
    def create_ticker_from_binance_item(cls, ticker_item):
        return cls(ticker_name=ticker_item["symbol"], price=ticker_item["price"])
