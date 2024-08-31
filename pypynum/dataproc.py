from .types import Any, Callable, Union, real


class Series:
    def __init__(self, data: Any = None, index: Any = None) -> None:
        if data is None:
            self.data = []
            self.index = []
        else:
            self.data = list(data)
            self.index = list(index)

    def __repr__(self) -> str:
        def string(obj):
            return obj.__str__().__repr__()[1:-1]

        escaped_indices = tuple(map(string, self.index))
        escaped_data = tuple(map(string, self.data))
        max_index_length = max(map(len, escaped_indices))
        max_data_length = max(map(len, escaped_data))
        format_str = "{{:<{}}}    {{:>{}}}".format(max_index_length, max_data_length)
        return "\n".join([format_str.format(index, data) for index, data in zip(escaped_indices, escaped_data)])

    def describe(self, percentiles: Any = (0.25, 0.5, 0.75), interpolation: str = "linear", dof: int = 1) -> "Series":
        from .maths import quantile
        data = self.__dropna(None)
        if all([isinstance(x, (int, float)) for x in self.data]):
            count = len(data)
            mean = sum(data) / count
            std = (sum([(x - mean) ** 2 for x in data]) / (count - dof)) ** 0.5
            min_val = min(data)
            max_val = max(data)
            percentiles = sorted(set(percentiles))

            def func(q):
                return quantile(data, q, interpolation, True)

            data = [count, mean, std, min_val] + list(map(func, percentiles)) + [max_val]
            index = ["count", "mean", "std", "min"] + [str(p * 100) + "%" for p in percentiles] + ["max"]
            return Series(data, index)
        else:
            count = len(data)
            unique = len(set(data))
            top = max(data, key=data.count)
            freq = data.count(top)
            data = [count, unique, top, freq]
            index = ["count", "unique", "top", "freq"]
            return Series(data, index)

    def count(self, skipna: bool = True) -> int:
        return len(self.__dropna(None) if skipna else self.data)

    def mean(self, skipna: bool = True) -> real:
        from .maths import mean
        data = self.__dropna(None) if skipna else self.data
        return mean(data)

    def var(self, dof: int = 1, skipna: bool = True) -> real:
        from .maths import var
        data = self.__dropna(None) if skipna else self.data
        return var(data, dof)

    def std(self, dof: int = 1, skipna: bool = True) -> real:
        from .maths import std
        data = self.__dropna(None) if skipna else self.data
        return std(data, dof)

    def min(self, skipna: bool = True) -> real:
        data = self.__dropna(None) if skipna else self.data
        return min(data)

    def max(self, skipna: bool = True) -> real:
        data = self.__dropna(None) if skipna else self.data
        return max(data)

    def quantile(self, q: real, interpolation: str = "linear", skipna: bool = True) -> real:
        from .maths import quantile
        data = self.__dropna(None) if skipna else self.data
        return quantile(data, q, interpolation)

    def __dropna(self, ignore_index: Union[bool, None] = False) -> Union["Series", tuple]:
        filtered_pairs = [(x, idx) for x, idx in zip(self.data, self.index) if self.__notna(x)]
        new_data, new_index = zip(*filtered_pairs) if filtered_pairs else ((), ())
        if ignore_index is None:
            return new_data
        if ignore_index:
            new_index = range(len(new_data))
        return Series(new_data, new_index)

    def dropna(self, ignore_index: bool = False) -> "Series":
        return self.__dropna(bool(ignore_index))

    def __getattribute__(self, item: Any) -> Any:
        if item == "data" or item == "index":
            return list(super().__getattribute__(item))
        else:
            if item in self.index:
                data = [x for x, idx in zip(self.data, self.index) if idx == item]
                length = len(data)
                if length == 1:
                    return data[0]
                else:
                    return Series(data, [item for _ in range(length)])
            else:
                return super().__getattribute__(item)

    def __setattr__(self, key: Any, value: Any) -> None:
        if key == "data":
            if isinstance(value, (list, tuple)):
                if hasattr(self, "index") and len(value) != len(self.index):
                    raise ValueError("Length of 'data' does not match length of 'index'.")
                super().__setattr__(key, list(value))
            else:
                raise TypeError("'data' must be a list or tuple.")
        elif key == "index":
            if value is None:
                super().__setattr__(key, list(range(len(self.data))))
            elif isinstance(value, (list, tuple)):
                if len(self.data) != len(value):
                    raise ValueError("Length of 'index' does not match length of 'data'.")
                super().__setattr__(key, list(value))
            else:
                raise TypeError("'index' must be a list or tuple.")
        else:
            if key in self.index:
                data = [x if idx != key else value for x, idx in zip(self.data, self.index)]
                super().__setattr__("data", data)
            else:
                super().__setattr__(key, value)

    def apply(self, func: Callable) -> "Series":
        return Series(map(func, self.data), self.index)

    def head(self, n: int = 5) -> "Series":
        return Series(self.data[:n], self.index[:n])

    def tail(self, n: int = 5) -> "Series":
        return Series(self.data[-n:], self.index[-n:])

    def abs(self) -> "Series":
        return Series(map(abs, self.data), self.index)

    def sum(self, skipna: bool = True) -> real:
        from math import fsum
        data = self.__dropna(None) if skipna else self.data
        return fsum(data)

    def idxmax(self, skipna: bool = True) -> Any:
        data = self.__dropna(None) if skipna else self.data
        return self.index[data.index(max(data))]

    def idxmin(self, skipna: bool = True) -> Any:
        data = self.__dropna(None) if skipna else self.data
        return self.index[data.index(min(data))]

    def isna(self) -> "Series":
        return Series(map(self.__isna, self.data), self.index)

    def notna(self) -> "Series":
        return Series(map(self.__notna, self.data), self.index)

    def fillna(self, value: Any) -> "Series":
        filled_data = [value if self.__isna(x) else x for x in self.data]
        return Series(filled_data, self.index)

    def replace(self, to_replace: Any, value: Any) -> "Series":
        return Series([value if x == to_replace else x for x in self.data], self.index)

    def copy(self) -> "Series":
        return Series(self.data, self.index)

    def __sort(self, items, ascending=True, na_position="last"):
        def sort_key(item):
            value = item[1]
            isna = self.__isna(value)
            return isna if bool(ascending) is (na_position == "last") else not isna, item[0] if isna else value

        if na_position not in ["first", "last"]:
            raise ValueError("'na_position' must be either 'first' or 'last'")
        sorted_items = sorted(enumerate(items), key=sort_key, reverse=not ascending)
        return tuple(zip(*sorted_items))

    def sort_index(self, ascending: bool = True, na_position: str = "last", ignore_index: bool = False) -> "Series":
        sorted_positions, sorted_indices = self.__sort(self.index, ascending, na_position)
        sorted_values = map(self.data.__getitem__, sorted_positions)
        new_index = range(len(self.index)) if ignore_index else sorted_indices
        return Series(sorted_values, new_index)

    def sort_values(self, ascending: bool = True, na_position: str = "last", ignore_index: bool = False) -> "Series":
        sorted_positions, sorted_values = self.__sort(self.data, ascending, na_position)
        new_index = range(len(self.index)) if ignore_index else map(self.index.__getitem__, sorted_positions)
        return Series(sorted_values, new_index)

    @staticmethod
    def __isna(value: Any) -> bool:
        return value is None or isinstance(value, float) and value != value

    @staticmethod
    def __notna(value: Any) -> bool:
        return value is not None and isinstance(value, float) and value == value

    def cov(self, other: "Series", min_periods: int = None, dof=1) -> real:
        from .maths import cov
        data1 = self.__dropna(None)
        data2 = other.__dropna(None)
        if len(data1) < min_periods or len(data2) < min_periods:
            return float("nan")
        return cov(data1, data2, dof)

    def corr(self, other: "Series", min_periods: int = None) -> real:
        from .maths import corr_coeff
        data1 = self.__dropna(None)
        data2 = other.__dropna(None)
        if len(data1) < min_periods or len(data2) < min_periods:
            return float("nan")
        return corr_coeff(data1, data2)
