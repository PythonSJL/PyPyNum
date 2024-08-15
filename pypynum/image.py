import warnings
import zlib
from .Matrix import mat

MODE = {"GRAY": 0, "RGB": 2, "INDEX": 3, "GRAYA": 4, "RGBA": 6}
NUMBER = {0: "GRAY", 2: "RGB", 3: "INDEX", 4: "GRAYA", 6: "RGBA"}
TABLE = [
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
        init = (TABLE[c & 0xff] ^ (init >> 8)) & 4294967295
    return init ^ xor


class PNG:
    """
    Introduction
    ==========
    This is a PNG class written in pure Python,
    supporting the creation, reading, modification, and saving of PNG images.

    Roadmap
    ==========
    This class is currently in development.
    Future updates will expand its functionality to enhance the capabilities of working with PNG images.

    Usage
    ==========

    Creating a new PNG image:
    ----------
    To create a new PNG image, instantiate the PNG class and use the `new` method
    to define the image dimensions, bit depth, and color mode.

    - from png import PNG

    # Create a new image with a width of 200 pixels, a height of 100 pixels,
    an 8-bit depth, and the default RGB color mode.

    - image = PNG()
    - image.new(200, 100, 8)

    # Optionally, you can specify a background color. For example, to create a new image with a blue background:

    - image.new(200, 100, 8, color=(0, 0, 255))

    Reading an existing PNG image:
    ----------
    To read an existing PNG image from a file, use the `read` method.

    - image = PNG()
    - image.read("example.png")

    Modifying a pixel:
    ----------
    To modify the color of a pixel at a specific coordinate, use the `setp` method.

    # Set the pixel at (10, 10) to red (255, 0, 0).

    - image.setp(10, 10, (255, 0, 0))

    Getting a pixel's color:
    ----------
    To retrieve the color of a pixel at a specific coordinate, use the `getp` method.

    - color = image.getp(10, 10)
    - print(color)  # Output: (255, 0, 0) for the example above

    Saving the image:
    ----------
    To save the image to a file, use the `write` method.

    - image.write("output.png")

    Getting image information:
    ----------
    To obtain information about the image, such as its dimensions and color mode, use the `info` method.

    - info = image.info()
    - print(info)  # Output: {'width': 200, 'height': 100, 'bit_depth': 8, 'color_mode': 'RGB'}
    """

    def __init__(self) -> None:
        self.__bit_depth = None
        self.__color_mode = None
        self.__width = None
        self.__height = None
        self.__pixels = None

    def new(self, width: int, height: int, bit_depth: int, color: tuple = (), color_mode: str = "RGB") -> None:
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
        self.__width = width
        self.__height = height
        self.__pixels = mat([[color for _ in range(width)] for _ in range(height)])
        self.__bit_depth = bit_depth
        self.__color_mode = color_mode

    def read(self, filename: str) -> None:
        with open(filename, "rb") as rb:
            part = rb.read(8)
            if part != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Not a PNG file")
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
            self.__width, self.__height = int.from_bytes(width, "big"), int.from_bytes(height, "big")
            part = rb.read(5)
            temp.append(width)
            temp.append(height)
            temp.append(part)
            self.__bit_depth = part[0]
            if self.__bit_depth not in [8, 16]:
                warnings.warn("Only supports parsing PNG files with a depth of 8 or 16 bits and true color",
                              RuntimeWarning)
            self.__color_mode = NUMBER[part[1]]
            part = rb.read(4)
            if crc(b"".join(temp)) != int.from_bytes(part, "big"):
                warnings.warn("IHDR chunk verification failed", RuntimeWarning)
            while True:
                part = rb.read(4)
                length = int.from_bytes(part, "big")
                part = rb.read(4)
                if part == b"IDAT":
                    pixels = rb.read(length)
                    break
                else:
                    rb.read(length)
                    rb.read(4)
            part = rb.read(4)
            if crc(b"IDAT" + pixels) != int.from_bytes(part, "big"):
                warnings.warn("IDAT chunk verification failed", RuntimeWarning)
            while True:
                part = rb.read(4)
                length = int.from_bytes(part, "big")
                part = rb.read(4)
                if part == b"IEND":
                    part = rb.read(4)
                    if part != b"\xaeB`\x82":
                        warnings.warn("IEND chunk verification failed", RuntimeWarning)
                    break
                else:
                    rb.read(length)
                    rb.read(4)
            de = zlib.decompress(pixels)
            length = len(de)
            line = length // self.__height
            step = len(self.__color_mode)
            if self.__bit_depth == 8:
                self.__pixels = mat([[tuple(de[y + x:y + x + step]) for x in range(1, line, step)]
                                     for y in range(0, length, line)])
            elif self.__bit_depth == 16:
                step *= 2
                self.__pixels = mat([[tuple([int.from_bytes(de[y + x + i:y + x + i + 2], "big")
                                             for i in range(0, step, 2)]) for x in range(1, line, step)]
                                     for y in range(0, length, line)])

    def write(self, filename: str = None) -> bytes:
        if self.__pixels is None:
            raise ValueError("Please use the 'new' method to create an image first")
        ihdr = b"".join([b"IHDR", self.__width.to_bytes(4, "big"), self.__height.to_bytes(4, "big"),
                         bytes([self.__bit_depth]), bytes([MODE[self.__color_mode]]), b"\x00\x00\x00"])

        def func(x):
            return b"".join([_.to_bytes(self.__bit_depth // 8, "big") for _ in x])

        if self.__bit_depth == 8:
            func = bytes
        idat = b"IDAT" + zlib.compress(b"".join([b"\x00" + b"".join(map(func, row)) for row in self.__pixels]))
        png = b"".join([
            b"\x89PNG\r\n\x1a\n", b"\x00\x00\x00\r", ihdr, crc(ihdr).to_bytes(4, "big"),
            b"\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9", b"\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05",
            b"\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d",
            (len(idat) - 4).to_bytes(4, "big"), idat, crc(idat).to_bytes(4, "big"),
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        ])
        if filename:
            with open(filename, "wb") as wb:
                wb.write(png)
        else:
            return png

    def getp(self, x: int, y: int) -> tuple:
        return self.__pixels[y, x]

    def setp(self, x: int, y: int, color: tuple) -> None:
        color = tuple(color)
        if len(color) != len(self.__color_mode):
            raise ValueError("The specified color can only be an {} tuple".format(self.__color_mode))
        maximum = (1 << self.__bit_depth) - 1
        if not all([isinstance(_, int) and 0 <= _ <= maximum for _ in color]):
            raise ValueError("The color value must be an integer between 0 and {} when the bit depth is {}".format(
                maximum, self.__bit_depth))
        self.__pixels[y, x] = color

    def info(self) -> dict:
        return {"width": self.__width, "height": self.__height,
                "bit_depth": self.__bit_depth, "color_mode": self.__color_mode}

    def __repr__(self) -> str:
        return self.__class__.__name__ + str(sorted(self.info().items())).replace("',", " =").replace("'", "")
