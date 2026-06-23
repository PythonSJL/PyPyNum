class FT1D:
    def __init__(self, *data):
        if not all([isinstance(i, (int, float, complex)) for i in data]):
            raise ValueError("The input data must all be numerical values")
        n = len(data)
        if n & (n - 1):
            n = 2 ** n.bit_length()
            data = list(data) + [0] * (n - len(data))
        self.data = data

    def __repr__(self):
        return self.__class__.__name__ + str(self.data)

    def fft(self):
        from cmath import exp

        def inner(data):
            n = len(data)
            if n <= 1:
                return data
            even = inner(data[0::2])
            odd = inner(data[1::2])
            t = [exp(-6.283185307179586j * k / n) * odd[k] for k in range(n // 2)]
            return [even[k] + t[k] for k in range(n // 2)] + [even[k] - t[k] for k in range(n // 2)]

        self.data = inner(self.data)

    def ifft(self):
        from cmath import exp

        def inner(data):
            n = len(data)
            if n <= 1:
                return data
            even = inner(data[0::2])
            odd = inner(data[1::2])
            t = [exp(6.283185307179586j * k / n) * odd[k] for k in range(n // 2)]
            return [even[k] + t[k] for k in range(n // 2)] + [even[k] - t[k] for k in range(n // 2)]

        self.data = [k / len(self.data) for k in inner(self.data)]
