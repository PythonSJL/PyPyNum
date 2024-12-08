import warnings
import zlib
from .matrices import mat

_CHANNEL = {"GRAY": 1, "RGB": 3, "INDEX": 1, "GRAYA": 2, "RGBA": 4}
_DEPTH = {8: ("INDEX", 8), 24: ("RGB", 8), 32: ("RGBA", 8), 48: ("RGB", 16), 64: ("RGBA", 16)}
_MODE = {"GRAY": 0, "RGB": 2, "INDEX": 3, "GRAYA": 4, "RGBA": 6}
_NUMBER = {0: "GRAY", 2: "RGB", 3: "INDEX", 4: "GRAYA", 6: "RGBA"}
_TABLE = [
    0, 1996959894, 3993919788, 2567524794, 124634137, 1886057615, 3915621685, 2657392035,
    249268274, 2044508324, 3772115230, 2547177864, 162941995, 2125561021, 3887607047, 2428444049,
    498536548, 1789927666, 4089016648, 2227061214, 450548861, 1843258603, 4107580753, 2211677639,
    325883990, 1684777152, 4251122042, 2321926636, 335633487, 1661365465, 4195302755, 2366115317,
    997073096, 1281953886, 3579855332, 2724688242, 1006888145, 1258607687, 3524101629, 2768942443,
    901097722, 1119000684, 3686517206, 2898065728, 853044451, 1172266101, 3705015759, 2882616665,
    651767980, 1373503546, 3369554304, 3218104598, 565507253, 1454621731, 3485111705, 3099436303,
    671266974, 1594198024, 3322730930, 2970347812, 795835527, 1483230225, 3244367275, 3060149565,
    1994146192, 31158534, 2563907772, 4023717930, 1907459465, 112637215, 2680153253, 3904427059,
    2013776290, 251722036, 2517215374, 3775830040, 2137656763, 141376813, 2439277719, 3865271297,
    1802195444, 476864866, 2238001368, 4066508878, 1812370925, 453092731, 2181625025, 4111451223,
    1706088902, 314042704, 2344532202, 4240017532, 1658658271, 366619977, 2362670323, 4224994405,
    1303535960, 984961486, 2747007092, 3569037538, 1256170817, 1037604311, 2765210733, 3554079995,
    1131014506, 879679996, 2909243462, 3663771856, 1141124467, 855842277, 2852801631, 3708648649,
    1342533948, 654459306, 3188396048, 3373015174, 1466479909, 544179635, 3110523913, 3462522015,
    1591671054, 702138776, 2966460450, 3352799412, 1504918807, 783551873, 3082640443, 3233442989,
    3988292384, 2596254646, 62317068, 1957810842, 3939845945, 2647816111, 81470997, 1943803523,
    3814918930, 2489596804, 225274430, 2053790376, 3826175755, 2466906013, 167816743, 2097651377,
    4027552580, 2265490386, 503444072, 1762050814, 4150417245, 2154129355, 426522225, 1852507879,
    4275313526, 2312317920, 282753626, 1742555852, 4189708143, 2394877945, 397917763, 1622183637,
    3604390888, 2714866558, 953729732, 1340076626, 3518719985, 2797360999, 1068828381, 1219638859,
    3624741850, 2936675148, 906185462, 1090812512, 3747672003, 2825379669, 829329135, 1181335161,
    3412177804, 3160834842, 628085408, 1382605366, 3423369109, 3138078467, 570562233, 1426400815,
    3317316542, 2998733608, 733239954, 1555261956, 3268935591, 3050360625, 752459403, 1541320221,
    2607071920, 3965973030, 1969922972, 40735498, 2617837225, 3943577151, 1913087877, 83908371,
    2512341634, 3803740692, 2075208622, 213261112, 2463272603, 3855990285, 2094854071, 198958881,
    2262029012, 4057260610, 1759359992, 534414190, 2176718541, 4139329115, 1873836001, 414664567,
    2282248934, 4279200368, 1711684554, 285281116, 2405801727, 4167216745, 1634467795, 376229701,
    2685067896, 3608007406, 1308918612, 956543938, 2808555105, 3495958263, 1231636301, 1047427035,
    2932959818, 3654703836, 1088359270, 936918000, 2847714899, 3736837829, 1202900863, 817233897,
    3183342108, 3401237130, 1404277552, 615818150, 3134207493, 3453421203, 1423857449, 601450431,
    3009837614, 3294710456, 1567103746, 711928724, 3020668471, 3272380065, 1510334235, 755167117
]


def crc(data, length=None, init=4294967295, xor=4294967295):
    if length is None:
        length = len(data)
    for n in range(length):
        c = init ^ data[n]
        init = (_TABLE[c & 0xff] ^ (init >> 8)) & 4294967295
    return init ^ xor


class BaseImage:
    def __init__(self) -> None:
        self._bit_depth = None
        self._color_mode = None
        self._width = None
        self._height = None
        self._pixels = None

    @property
    def bit_depth(self) -> int:
        return self._bit_depth

    @property
    def color_mode(self) -> str:
        return self._color_mode

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def info(self) -> dict:
        return {"width": self._width, "height": self._height,
                "bit_depth": self._bit_depth, "color_mode": self._color_mode}

    def new(self, width: int, height: int, color: tuple = (), color_mode: str = "RGB", bit_depth: int = 8) -> None:
        color_mode = color_mode.strip().upper()
        supported = ["RGB", "RGBA"]
        maximum = (1 << bit_depth) - 1
        if color_mode not in supported:
            raise ValueError("The current version supports only one color mode from {}".format(
                ", ".join(map(repr, supported))))
        if bit_depth not in [8, 16]:
            raise ValueError("The bit depth of a single channel must be 8 or 16")
        if color:
            color = tuple(color)
        else:
            color = (maximum,) * len(color_mode)
        if len(color) != len(color_mode):
            raise ValueError("The specified color can only be an {} tuple".format(color_mode))
        if not all([isinstance(_, int) and 0 <= _ <= maximum for _ in color]):
            raise ValueError("The color value must be an integer between 0 and {} when the bit depth is {}".format(
                maximum, bit_depth))
        self._width = width
        self._height = height
        self._pixels = mat([[color for _ in range(width)] for _ in range(height)])
        self._bit_depth = bit_depth
        self._color_mode = color_mode

    def __getitem__(self, item: tuple) -> tuple:
        x, y = item
        return self._pixels[y, x]

    def __setitem__(self, key: tuple, value: tuple) -> None:
        x, y = key
        color = tuple(value)
        if len(color) != _CHANNEL[self._color_mode]:
            raise ValueError("The specified color can only be an {} tuple".format(self._color_mode))
        maximum = (1 << self._bit_depth) - 1
        if not all([isinstance(_, int) and 0 <= _ <= maximum for _ in color]):
            raise ValueError("The color value must be an integer between 0 and {} when the bit depth is {}".format(
                maximum, self._bit_depth))
        self._pixels[y, x] = color

    def __repr__(self) -> str:
        return self.__class__.__name__ + str(sorted(self.info().items())).replace("',", ":").replace("'", "")


class BMP(BaseImage):
    def new(self, width: int, height: int, color: tuple = (), color_mode: str = "RGB", bit_depth: int = 8) -> None:
        if bit_depth > 8:
            raise NotImplementedError("The bit_depth is too high for this implementation")
        if color:
            color = tuple(color)
        else:
            color = ((1 << bit_depth) - 1,) * len(color_mode)
        r, g, b, *a = color
        color = (b, g, r, a[0]) if a else (b, g, r)
        super().new(width, height, color, color_mode, bit_depth)

    def read(self, filename: str) -> None:
        with open(filename, "rb") as rb:
            if rb.read(2) != b"BM":
                raise ValueError("The file is not a valid BMP image")
            rb.read(12)
            header_size = int.from_bytes(rb.read(4), "little")
            if header_size != 40:
                raise ValueError("The BMP image has an invalid header size of {} bytes".format(header_size))
            width, height = int.from_bytes(rb.read(4), "little"), int.from_bytes(rb.read(4), "little")
            planes = int.from_bytes(rb.read(2), "little")
            bits_per_pixel = int.from_bytes(rb.read(2), "little")
            compression = int.from_bytes(rb.read(4), "little")
            rb.read(20)
            if planes != 1:
                raise ValueError("The BMP image must have a single plane of color data, "
                                 "but found {} planes".format(planes))
            if compression != 0:
                raise ValueError("The BMP image must not be compressed, "
                                 "but found compression type {}".format(compression))
            if bits_per_pixel not in _DEPTH:
                raise ValueError("The BMP image has an unsupported color depth of {} bits".format(bits_per_pixel))
            self._color_mode, self._bit_depth = _DEPTH[bits_per_pixel]
            palette_size = 1 << bits_per_pixel if bits_per_pixel <= 8 else 0
            if palette_size:
                raise NotImplementedError
            row_size = (width * bits_per_pixel + 31 & -32) >> 3
            pixels = rb.read(row_size * height)
            if bits_per_pixel <= 8:
                raise NotImplementedError
            else:
                length = _CHANNEL[self._color_mode]
                self._pixels = mat([[tuple(pixels[row * row_size + i:row * row_size + i + length])
                                     for i in range(0, width * length, length)] for row in range(height - 1, -1, -1)])
            self._width, self._height = width, height

    def write(self, filename: str = None) -> bytes:
        if self._pixels is None:
            raise ValueError("Please create or read an image first")
        channel_size = _CHANNEL[self._color_mode]
        filled_width = (self._width * self._bit_depth * channel_size + 31 & -32) >> 3
        filler = b"\x00" * (filled_width - self._width * channel_size)
        pixels = b"".join([b"".join(map(bytes, row)) + filler for row in self._pixels][::-1])
        pixel_size = filled_width * self._height * channel_size
        palette_size = 1 << self._bit_depth if self._color_mode == "INDEX" else 0
        image = b"".join([b"BM", (pixel_size + palette_size + 54).to_bytes(4, "little"), b"\x00" * 4,
                          (palette_size + 54).to_bytes(4, "little"), b"(\x00\x00\x00",
                          self._width.to_bytes(4, "little"), self._height.to_bytes(4, "little"), b"\x01\x00",
                          (channel_size * self._bit_depth).to_bytes(2, "little"), b"\x00" * 24, pixels])
        if filename:
            with open(filename, "wb") as wb:
                wb.write(image)
        else:
            return image

    def __getitem__(self, item: tuple) -> tuple:
        x, y = item
        color = self._pixels[y, x]
        b, g, r, *a = color
        return (r, g, b, a[0]) if a else (r, g, b)

    def __setitem__(self, key: tuple, value: tuple) -> None:
        x, y = key
        color = tuple(value)
        if len(color) != _CHANNEL[self._color_mode]:
            raise ValueError("The specified color can only be an {} tuple".format(self._color_mode))
        r, g, b, *a = color
        color = (b, g, r, a[0]) if a else (b, g, r)
        maximum = (1 << self._bit_depth) - 1
        if not all([isinstance(_, int) and 0 <= _ <= maximum for _ in color]):
            raise ValueError("The color value must be an integer between 0 and {} when the bit depth is {}".format(
                maximum, self._bit_depth))
        self._pixels[y, x] = color


class PNG(BaseImage):
    def read(self, filename: str) -> None:
        with open(filename, "rb") as rb:
            part = rb.read(8)
            if part != b"\x89PNG\r\n\x1a\n":
                raise ValueError("The file is not a valid PNG image")
            temp = []
            part = rb.read(4)
            length = int.from_bytes(part, "big")
            part = rb.read(4)
            temp.append(part)
            if part != b"IHDR":
                raise ValueError("Missing IHDR chunk")
            if length != 13:
                raise ValueError("The length of IHDR chunk must be 13")
            width, height = rb.read(4), rb.read(4)
            self._width, self._height = int.from_bytes(width, "big"), int.from_bytes(height, "big")
            part = rb.read(5)
            temp.append(width)
            temp.append(height)
            temp.append(part)
            self._bit_depth = part[0]
            if self._bit_depth not in [8, 16]:
                warnings.warn("Current version only supports parsing PNG files with each channel "
                              "at 8 or 16 bits of depth and true color", RuntimeWarning)
            self._color_mode = _NUMBER[part[1]]
            part = rb.read(4)
            if crc(b"".join(temp)) != int.from_bytes(part, "big"):
                warnings.warn("IHDR chunk verification failed", RuntimeWarning)
            pixels = []
            while True:
                part = rb.read(4)
                length = int.from_bytes(part, "big")
                part = rb.read(4)
                if part == b"IDAT":
                    temp = rb.read(length)
                    pixels.append(temp)
                    part = rb.read(4)
                    if crc(b"IDAT" + temp) != int.from_bytes(part, "big"):
                        warnings.warn("IDAT chunk verification failed for chunk number {}".format(len(pixels)),
                                      RuntimeWarning)
                elif part == b"IEND":
                    part = rb.read(4)
                    if part != b"\xaeB`\x82":
                        warnings.warn("IEND chunk verification failed", RuntimeWarning)
                    break
                else:
                    rb.read(length + 4)
            de = zlib.decompressobj().decompress(b"".join(pixels))
            length = len(de)
            line = length // self._height
            step = _CHANNEL[self._color_mode]
            if self._bit_depth == 8:
                self._pixels = mat([[tuple(de[y + x:y + x + step]) for x in range(1, line, step)]
                                    for y in range(0, length, line)])
            elif self._bit_depth == 16:
                step *= 2
                self._pixels = mat([[tuple([int.from_bytes(de[y + x + i:y + x + i + 2], "big")
                                            for i in range(0, step, 2)]) for x in range(1, line, step)]
                                    for y in range(0, length, line)])

    def write(self, filename: str = None) -> bytes:
        if self._pixels is None:
            raise ValueError("Please create or read an image first")
        ihdr = b"".join([b"IHDR", self._width.to_bytes(4, "big"), self._height.to_bytes(4, "big"),
                         bytes([self._bit_depth]), bytes([_MODE[self._color_mode]]), b"\x00\x00\x00"])
        length = self._bit_depth // 8

        def func(x):
            return b"".join([_.to_bytes(length, "big") for _ in x])

        if self._bit_depth == 8:
            func = bytes
        idat = b"IDAT" + zlib.compress(b"".join([b"\x00" + b"".join(map(func, row)) for row in self._pixels]))
        image = b"".join([
            b"\x89PNG\r\n\x1a\n", b"\x00\x00\x00\r", ihdr, crc(ihdr).to_bytes(4, "big"),
            b"\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9", b"\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05",
            b"\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d",
            (len(idat) - 4).to_bytes(4, "big"), idat, crc(idat).to_bytes(4, "big"),
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        ])
        if filename:
            with open(filename, "wb") as wb:
                wb.write(image)
        else:
            return image
