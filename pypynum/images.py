import collections
import math
import os
import warnings
import zlib
from .kernels import matmul8x8kernel
from .types import Any, arr

_CHANNEL = {"GRAY": 1, "RGB": 3, "INDEX": 1, "GRAYA": 2, "RGBA": 4}
_DEPTH = {8: ("INDEX", 8), 24: ("RGB", 8), 32: ("RGBA", 8), 48: ("RGB", 16), 64: ("RGBA", 16)}
_MODE = {"GRAY": 0, "RGB": 2, "INDEX": 3, "GRAYA": 4, "RGBA": 6}
_NUMBER = {0: "GRAY", 2: "RGB", 3: "INDEX", 4: "GRAYA", 6: "RGBA"}
_STD_LUMA_QTABLE = [[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]]
_STD_CHROMA_QTABLE = [[17, 18, 24, 47, 99, 99, 99, 99],
                      [18, 21, 26, 66, 99, 99, 99, 99],
                      [24, 26, 56, 99, 99, 99, 99, 99],
                      [47, 66, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99]]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "huffman"), "r") as __r:
    __AC, __IAC = eval(__r.read())


def rgb2ycbcr(weights: arr) -> tuple:
    r, g, b, *_ = weights
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168735892 * r - 0.331264108 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418687589 * g - 0.081312411 * b + 128
    return round(y), round(cb), round(cr)


def ycbcr2rgb(weights: arr) -> tuple:
    y, cb, cr, *_ = weights
    cb = cb - 128
    cr = cr - 128
    r = y + 1.402 * cr
    g = y - 0.344136286 * cb - 0.714136286 * cr
    b = y + 1.772 * cb
    return round(r), round(g), round(b)


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
        supported = ("RGB", "RGBA")
        maximum = (1 << bit_depth) - 1
        if color_mode not in supported:
            raise ValueError("The current version supports only one color mode from {}".format(
                ", ".join(map(repr, supported))))
        if bit_depth not in (8, 16):
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
        bytes_per_value = bit_depth // 8
        pixel_bytes = b""
        for c in color:
            pixel_bytes += c.to_bytes(bytes_per_value, "big")
        row_bytes = pixel_bytes * width
        self._pixels = [bytearray(row_bytes) for _ in range(height)]
        self._bit_depth = bit_depth
        self._color_mode = color_mode

    def __getitem__(self, item: tuple) -> tuple:
        x, y = item
        channel_count = _CHANNEL[self._color_mode]
        bytes_per_value = self._bit_depth // 8
        pixel_size = channel_count * bytes_per_value
        row_bytes = self._pixels[y]
        offset = x * pixel_size
        pixel_data = row_bytes[offset: offset + pixel_size]
        if self._bit_depth == 8:
            return tuple(pixel_data)
        else:
            return tuple([int.from_bytes(pixel_data[i:i + bytes_per_value], "big")
                          for i in range(0, pixel_size, bytes_per_value)])

    def __setitem__(self, key: tuple, value: tuple) -> None:
        x, y = key
        color = tuple(value)
        if len(color) != _CHANNEL[self._color_mode]:
            raise ValueError("The specified color can only be an {} tuple".format(self._color_mode))
        maximum = (1 << self._bit_depth) - 1
        if not all([isinstance(_, int) and 0 <= _ <= maximum for _ in color]):
            raise ValueError("The color value must be an integer between 0 and {} when the bit depth is {}".format(
                maximum, self._bit_depth))
        channel_count = _CHANNEL[self._color_mode]
        bytes_per_value = self._bit_depth // 8
        pixel_size = channel_count * bytes_per_value
        offset = x * pixel_size
        current_row = self._pixels[y]
        for i, val in enumerate(value):
            start = offset + i * bytes_per_value
            current_row[start: start + bytes_per_value] = val.to_bytes(bytes_per_value, "big")

    def __repr__(self) -> str:
        return self.__class__.__name__ + str(sorted(self.info().items())).replace("',", ":").replace("'", "")


class BMP(BaseImage):
    def new(self, width: int, height: int, color: tuple = (), color_mode: str = "RGB", bit_depth: int = 8) -> None:
        if bit_depth > 8:
            raise NotImplementedError("The bit_depth is too high for this implementation")
        super().new(width, height, color, color_mode, bit_depth)

    @staticmethod
    def rgb2bgr(color: tuple) -> tuple:
        r, g, b, *a = color
        return (b, g, r, a[0]) if a else (b, g, r)

    @staticmethod
    def bgr2rgb(color: tuple) -> tuple:
        b, g, r, *a = color
        return (r, g, b, a[0]) if a else (r, g, b)

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
                self._pixels = []
                for row in range(height - 1, -1, -1):
                    start = row * row_size
                    raw_row = pixels[start: start + width * length]
                    self._pixels.append(bytearray(b"".join([bytes(self.bgr2rgb(raw_row[i: i + length]))
                                                            for i in range(0, width * length, length)])))
            self._width, self._height = width, height

    def write(self, filename: str = None) -> bytes:
        if self._pixels is None:
            raise ValueError("Please create or read an image first")
        channel_size = _CHANNEL[self._color_mode]
        filled_width = (self._width * self._bit_depth * channel_size + 31 & -32) >> 3
        filler = b"\x00" * (filled_width - self._width * channel_size)
        pixel_data_list = []
        for row in self._pixels:
            bgr_row = [bytes(self.rgb2bgr(row[i:i + channel_size])) for i in range(0, len(row), channel_size)]
            bgr_row.append(filler)
            pixel_data_list.append(b"".join(bgr_row))
        pixels = b"".join(reversed(pixel_data_list))
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
            if self._bit_depth not in (8, 16):
                warnings.warn("Current version only supports parsing PNG files with each channel "
                              "at 8 or 16 bits of depth and true color", RuntimeWarning)
            self._color_mode = _NUMBER[part[1]]
            part = rb.read(4)
            if zlib.crc32(b"".join(temp)) != int.from_bytes(part, "big"):
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
                    if zlib.crc32(b"IDAT" + temp) != int.from_bytes(part, "big"):
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
                filtered_rows = []
                prev_row = None
                for y in range(0, length, line):
                    filter_type = de[y]
                    row_data = de[y + 1:y + line]
                    filtered_row = [row_data[x:x + step] for x in range(0, len(row_data), step)]
                    original_row = png_reverse_filter(filtered_row, prev_row, filter_type)
                    filtered_rows.append(original_row)
                    prev_row = original_row
                self._pixels = [bytearray(b"".join(map(bytes, row))) for row in filtered_rows]
            elif self._bit_depth == 16:
                step *= 2
                raw_rows = []
                for y in range(0, length, line):
                    raw_rows.append([[int.from_bytes(de[y + x + i:y + x + i + 2], "big")
                                      for i in range(0, step, 2)] for x in range(1, line, step)])
                self._pixels = [bytearray(b"".join([val.to_bytes(2, "big") for pixel in row for val in pixel]))
                                for row in raw_rows]

    def write(self, filename: str = None, apply_filter: str = "none") -> bytes:
        if self._pixels is None:
            raise ValueError("Please create or read an image first")
        ihdr = b"".join([b"IHDR", self._width.to_bytes(4, "big"), self._height.to_bytes(4, "big"),
                         bytes([self._bit_depth]), bytes([_MODE[self._color_mode]]), b"\x00\x00\x00"])
        length = self._bit_depth // 8

        def func(x):
            return b"".join([_.to_bytes(length, "big") for _ in x])

        filter_map = ("none", "sub", "up", "average", "paeth", "adaptive")
        apply_filter = apply_filter.strip().lower()
        if apply_filter not in filter_map:
            raise ValueError("apply_filter must be one of " + str(filter_map))
        if self._bit_depth != 8 and apply_filter != "none":
            warnings.warn("Filters are only supported for 8-bit images. Using \"none\" filter", RuntimeWarning)
            apply_filter = "none"
        pixel_matrix = []
        bpp = _CHANNEL[self._color_mode] * length
        for row in self._pixels:
            pixel_row = []
            for i in range(0, len(row), bpp):
                pixel = []
                for k in range(0, bpp, length):
                    val = int.from_bytes(row[i + k: i + k + length], "big")
                    pixel.append(val)
                pixel_row.append(pixel)
            pixel_matrix.append(pixel_row)
        precomputed = []
        num_rows = len(pixel_matrix)
        for r in range(num_rows):
            row_filters_data = []
            current_row = pixel_matrix[r]
            prev_row = pixel_matrix[r - 1] if r > 0 else None
            for f_type in range(5):
                filtered_pixels = png_apply_filter(current_row, prev_row, f_type)
                if self._bit_depth == 8:
                    row_bytes = b"".join(map(bytes, filtered_pixels))
                else:
                    row_bytes = b"".join(map(func, filtered_pixels))
                row_filters_data.append(row_bytes)
            precomputed.append(row_filters_data)
        candidates = []
        if apply_filter == "adaptive":
            mixed_rows = []
            for r in range(num_rows):
                if r == 0:
                    valid_filters = (0, 1)
                else:
                    valid_filters = (0, 1, 2, 3, 4)
                best_f = 0
                best_ent = float("inf")
                for f in valid_filters:
                    ent = entropy(precomputed[r][f])
                    if ent < best_ent:
                        best_ent = ent
                        best_f = f
                mixed_rows.append(bytes([best_f]) + precomputed[r][best_f])
            candidates.append(mixed_rows)
            for first_f in (0, 1):
                for other_f in range(5):
                    scheme_rows = []
                    for r in range(num_rows):
                        f = other_f if r else first_f
                        scheme_rows.append(bytes([f]) + precomputed[r][f])
                    candidates.append(scheme_rows)
        else:
            fixed_f = filter_map.index(apply_filter)
            fixed_rows = []
            for r in range(num_rows):
                fixed_rows.append(bytes([fixed_f]) + precomputed[r][fixed_f])
            candidates.append(fixed_rows)
        best = None
        min_size = float("inf")
        for scanlines in candidates:
            c_data = zlib.compress(b"".join(scanlines))
            size = len(c_data)
            if size < min_size:
                min_size = size
                best = c_data
        idat = b"IDAT" + best
        image = b"".join([
            b"\x89PNG\r\n\x1a\n",
            b"\x00\x00\x00\r",
            ihdr,
            zlib.crc32(ihdr).to_bytes(4, "big"),
            b"\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9",
            b"\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05",
            b"\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d",
            (len(idat) - 4).to_bytes(4, "big"),
            idat,
            zlib.crc32(idat).to_bytes(4, "big"),
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        ])
        if filename:
            with open(filename, "wb") as wb:
                wb.write(image)
        else:
            return image


def png_apply_filter(pixels: list, above_pixels: list = None, filter_type: int = 0) -> list:
    if filter_type == 0:
        return pixels
    channels = len(pixels[0]) if pixels else 0
    filtered_pixels = []
    for i in range(len(pixels)):
        filtered_pixel = []
        for c in range(channels):
            current_val = pixels[i][c]
            if filter_type == 1:
                left_val = pixels[i - 1][c] if i > 0 else 0
                filtered_val = (current_val - left_val) & 0xFF
            elif filter_type == 2:
                above_val = above_pixels[i][c] if above_pixels and i < len(above_pixels) else 0
                filtered_val = (current_val - above_val) & 0xFF
            elif filter_type == 3:
                left_val = pixels[i - 1][c] if i > 0 else 0
                above_val = above_pixels[i][c] if above_pixels and i < len(above_pixels) else 0
                filtered_val = (current_val - (left_val + above_val) // 2) & 0xFF
            elif filter_type == 4:
                left_val = pixels[i - 1][c] if i > 0 else 0
                above_val = above_pixels[i][c] if above_pixels and i < len(above_pixels) else 0
                above_left_val = above_pixels[i - 1][c] if above_pixels and 0 < i < len(above_pixels) else 0
                p = left_val + above_val - above_left_val
                pa = abs(p - left_val)
                pb = abs(p - above_val)
                pc = abs(p - above_left_val)
                if pa <= pb and pa <= pc:
                    predictor = left_val
                elif pb <= pc:
                    predictor = above_val
                else:
                    predictor = above_left_val
                filtered_val = (current_val - predictor) & 0xFF
            else:
                raise ValueError("Unknown filter type")
            filtered_pixel.append(filtered_val)
        filtered_pixels.append(filtered_pixel)
    return filtered_pixels


def png_reverse_filter(pixels: list, above_pixels: list = None, filter_type: int = 0) -> list:
    if filter_type == 0:
        return pixels
    channels = len(pixels[0]) if pixels else 0
    original_pixels = []
    for i in range(len(pixels)):
        original_pixel = []
        for c in range(channels):
            filtered_val = pixels[i][c]
            if filter_type == 1:
                left_val = original_pixels[i - 1][c] if i > 0 else 0
                original_val = (filtered_val + left_val) & 0xFF
            elif filter_type == 2:
                above_val = above_pixels[i][c] if above_pixels and i < len(above_pixels) else 0
                original_val = (filtered_val + above_val) & 0xFF
            elif filter_type == 3:
                left_val = original_pixels[i - 1][c] if i > 0 else 0
                above_val = above_pixels[i][c] if above_pixels and i < len(above_pixels) else 0
                original_val = (filtered_val + (left_val + above_val) // 2) & 0xFF
            elif filter_type == 4:
                left_val = original_pixels[i - 1][c] if i > 0 else 0
                above_val = above_pixels[i][c] if above_pixels and i < len(above_pixels) else 0
                above_left_val = above_pixels[i - 1][c] if above_pixels and 0 < i < len(above_pixels) else 0
                p = left_val + above_val - above_left_val
                pa = abs(p - left_val)
                pb = abs(p - above_val)
                pc = abs(p - above_left_val)
                if pa <= pb and pa <= pc:
                    predictor = left_val
                elif pb <= pc:
                    predictor = above_val
                else:
                    predictor = above_left_val
                original_val = (filtered_val + predictor) & 0xFF
            else:
                raise ValueError("Unknown filter type")
            original_pixel.append(original_val)
        original_pixels.append(original_pixel)
    return original_pixels


def entropy(data: Any) -> float:
    if not data:
        return 0.0
    values = collections.Counter(data).values()
    data_len = len(data)
    result = 0.0
    log2 = math.log2
    for count in values:
        p = count / data_len
        result -= p * log2(p)
    return result


def jpeg_dct8x8(block: arr, reverse: bool = False) -> list:
    dct8x8 = ((0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339),
              (0.49039264, 0.41573481, 0.27778512, 0.09754516, -0.09754516, -0.27778512, -0.41573481, -0.49039264),
              (0.46193977, 0.19134172, -0.19134172, -0.46193977, -0.46193977, -0.19134172, 0.19134172, 0.46193977),
              (0.41573481, -0.09754516, -0.49039264, -0.27778512, 0.27778512, 0.49039264, 0.09754516, -0.41573481),
              (0.35355339, -0.35355339, -0.35355339, 0.35355339, 0.35355339, -0.35355339, -0.35355339, 0.35355339),
              (0.27778512, -0.49039264, 0.09754516, 0.41573481, -0.41573481, -0.09754516, 0.49039264, -0.27778512),
              (0.19134172, -0.46193977, 0.46193977, -0.19134172, -0.19134172, 0.46193977, -0.46193977, 0.19134172),
              (0.09754516, -0.27778512, 0.41573481, -0.49039264, 0.49039264, -0.41573481, 0.27778512, -0.09754516))
    dct8x8t = ((0.35355339, 0.49039264, 0.46193977, 0.41573481, 0.35355339, 0.27778512, 0.19134172, 0.09754516),
               (0.35355339, 0.41573481, 0.19134172, -0.09754516, -0.35355339, -0.49039264, -0.46193977, -0.27778512),
               (0.35355339, 0.27778512, -0.19134172, -0.49039264, -0.35355339, 0.09754516, 0.46193977, 0.41573481),
               (0.35355339, 0.09754516, -0.46193977, -0.27778512, 0.35355339, 0.41573481, -0.19134172, -0.49039264),
               (0.35355339, -0.09754516, -0.46193977, 0.27778512, 0.35355339, -0.41573481, -0.19134172, 0.49039264),
               (0.35355339, -0.27778512, -0.19134172, 0.49039264, -0.35355339, -0.09754516, 0.46193977, -0.41573481),
               (0.35355339, -0.41573481, 0.19134172, 0.09754516, -0.35355339, 0.49039264, -0.46193977, 0.27778512),
               (0.35355339, -0.49039264, 0.46193977, -0.41573481, 0.35355339, -0.27778512, 0.19134172, -0.09754516))
    if reverse:
        dct8x8, dct8x8t = dct8x8t, dct8x8
    return matmul8x8kernel(matmul8x8kernel(dct8x8, block), dct8x8t)


def jpeg_zigzag(data: arr, reverse: bool = False) -> list:
    path = (0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31,
            40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63)
    if reverse:
        length = int(len(path) ** 0.5)
        return [[data[path[i * length + j]] for j in range(length)] for i in range(length)]
    result = [0] * len(path)
    for i, j in enumerate(path):
        result[j] = data[i // 8][i % 8]
    return result


def jpeg_rle_encoding(sequence: arr) -> list:
    if set(map(type, sequence)) != {int}:
        raise TypeError("All elements in the sequence must be integers")
    encoded = []
    i = 0
    length = len(sequence)
    while i != length:
        count = 0
        while i != length and count != 15 and sequence[i] == 0:
            count += 1
            i += 1
        if i != length:
            item = sequence[i]
            i += 1
            encoded.append((count, item))
    while encoded and encoded[-1] == (15, 0):
        del encoded[-1]
    if sequence[-1] == 0:
        encoded.append((0, 0))
    return encoded


def jpeg_rle_decoding(sequence: arr) -> list:
    if set(map(len, sequence)) != {2}:
        raise TypeError("All elements in the sequence must be tuples of length 2")
    decoded = []
    for pair in sequence:
        if pair == (0, 0):
            break
        count, item = pair
        if count:
            decoded.extend([0] * count)
        decoded.append(item)
    return decoded


def jpeg_adjust_qtable(qtable: arr, quality: int) -> list:
    if not 0 <= quality <= 100:
        raise ValueError("Quality must be between 0 and 100")
    rows = cols = 8
    if quality == 0:
        return [[255] * cols for _ in range(rows)]
    if quality == 100:
        return [[1] * cols for _ in range(rows)]
    factor = (100 - quality) / 50 if quality > 50 else 50 / quality
    return [[max(min(round(value * factor), 255), 1) for value in row] for row in qtable]


def jpeg_split_pixels(matrix: list) -> list:
    blocks = []
    block_size = 8
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    target_rows = ((rows + block_size - 1) // block_size) * block_size
    target_cols = ((cols + block_size - 1) // block_size) * block_size
    padded_matrix = []
    padding_width = target_cols - len(matrix[0])
    for r in matrix:
        padded_row = r[:]
        if padding_width > 0:
            padded_row += [padded_row[-1]] * padding_width
        padded_matrix.append(padded_row)
    padding_height = target_rows - len(padded_matrix)
    if padding_height > 0:
        last_row = padded_matrix[-1]
        padded_matrix += [last_row[:] for _ in range(padding_height)]
    for i in range(0, target_rows, block_size):
        row_block = padded_matrix[i:i + block_size]
        for j in range(0, target_cols, block_size):
            block = [row_block[k][j:j + block_size] for k in range(block_size)]
            blocks.append(block)
    return blocks


def jpeg_luma_dc_huff(data: Any, reverse: bool = False) -> Any:
    if reverse:
        fixed_codes = {"00": 0, "010": 1, "011": 2, "100": 3, "101": 4}
        if data in fixed_codes:
            return fixed_codes[data]
        data = list(data)
        endswith = data.pop()
        if endswith == "0" and len(data) >= 2:
            if set(data) == {"1"}:
                num_ones = len(data)
                return num_ones + 3
        return None
    if data >= 5:
        return "1" * (data - 3) + "0"
    else:
        return ("00", "010", "011", "100", "101")[data]


def jpeg_chroma_dc_huff(data: Any, reverse: bool = False) -> Any:
    if reverse:
        if data == "00":
            return 0
        if data == "01":
            return 1
        data = list(data)
        endswith = data.pop()
        if endswith == "0" and len(data) >= 1:
            if set(data) == {"1"}:
                num_ones = len(data)
                return num_ones + 1
        return None
    if data == 0:
        return "00"
    elif data == 1:
        return "01"
    else:
        return "1" * (data - 1) + "0"


def jpeg_category(data: Any, reverse: bool = False) -> Any:
    if reverse:
        if data == "":
            return 0
        length = len(data)
        value = int(data, 2)
        if data[0] == "0":
            value -= (1 << length) - 1
        return value
    if data == 0:
        return 0, ""
    length = data.bit_length()
    if data < 0:
        data += (1 << length) - 1
    return length, "{:0{}b}".format(data, length)


def jpeg_channel_encoding(matrix: list, quality: int, mode: int) -> tuple:
    def func(block):
        for i in range(8):
            for j in range(8):
                block[i][j] -= 128
        block = jpeg_dct8x8(block)
        for i in range(8):
            for j in range(8):
                block[i][j] = round(block[i][j] / qtable[i][j])
        return jpeg_zigzag(block)

    mode = bool(mode)
    blocks = jpeg_split_pixels(matrix)
    qtable = jpeg_adjust_qtable(_STD_CHROMA_QTABLE if mode else _STD_LUMA_QTABLE, quality)
    dpcm_values = []
    rle_values = []
    previous_pixel = None
    zigzags = map(func, blocks)
    for zigzag in zigzags:
        current_pixel = zigzag.pop(0)
        dpcm_values.append(current_pixel if previous_pixel is None else current_pixel - previous_pixel)
        rle_values.append(jpeg_rle_encoding(zigzag))
        previous_pixel = current_pixel
    encoded = [[(x, *jpeg_category(y)) for x, y in sublist] for sublist in rle_values]
    dc = jpeg_chroma_dc_huff if mode else jpeg_luma_dc_huff
    ac = __AC[mode]
    return ["".join((dc(length), value, "".join([ac[a, b] + c for a, b, c in sublist])))
            for (length, value), sublist in zip(map(jpeg_category, dpcm_values), encoded)], qtable


def jpeg_encode_pixels(pixels: arr, quality: int) -> tuple:
    split = tuple(map(list, zip(*[map(list, zip(*map(rgb2ycbcr, row))) for row in pixels])))
    encoded, qtables = zip(*[jpeg_channel_encoding(matrix, quality, mode) for mode, matrix in enumerate(split)])
    encoded, lqtable, cqtable = zip(*encoded), qtables[0], qtables[1]
    bin_str = "".join([item for sublist in encoded for item in sublist])
    bit_length = len(bin_str)
    if bit_length % 8 != 0:
        bin_str += "1" * (8 - bit_length % 8)
    return int(bin_str, 2).to_bytes(len(bin_str) // 8, "big").replace(b"\xff", b"\xff\x00"), lqtable, cqtable


def jpeg_decode_pixels(scan_data: bytes, lqtable: list, cqtable: list, width: int, height: int) -> list:
    bin_str = format(int.from_bytes(scan_data.replace(b"\xff\x00", b"\xff"), "big"), "b")
    bit_length = len(bin_str)
    if bit_length % 8 != 0:
        bin_str += "0" * (8 - bit_length % 8)
    bit_ptr = 0
    blocks_per_row = width // 8
    blocks_per_col = height // 8
    total_blocks = blocks_per_row * blocks_per_col
    channels = [[], [], []]
    prev_dc = [0, 0, 0]

    def read_bits(n):
        nonlocal bit_ptr
        if bit_ptr + n > len(bin_str):
            return None
        bits = bin_str[bit_ptr:bit_ptr + n]
        bit_ptr += n
        return bits

    def huffman_decode_ac(tree):
        nonlocal bit_ptr
        current_node = tree
        while True:
            if bit_ptr >= len(bin_str):
                return None
            bit = bin_str[bit_ptr]
            bit_ptr += 1
            if bit == "0":
                current_node = current_node[0]
            else:
                current_node = current_node[1]
            if isinstance(current_node[0], int):
                return current_node

    for _ in range(total_blocks):
        for channel_idx in range(3):
            qtable = lqtable if channel_idx == 0 else cqtable
            dc_huff_func = jpeg_chroma_dc_huff if channel_idx > 0 else jpeg_luma_dc_huff
            ac_tree = __IAC[1] if channel_idx > 0 else __IAC[0]
            for block_idx in range(total_blocks):
                dc_bits = ""
                while True:
                    if bit_ptr >= len(bin_str):
                        break
                    dc_bits += bin_str[bit_ptr]
                    bit_ptr += 1
                    dc_category = dc_huff_func(dc_bits, True)
                    if dc_category is None:
                        continue
                    else:
                        break
                dc_diff = 0
                if dc_category > 0:
                    bits = read_bits(dc_category)
                    if bits is None:
                        break
                    dc_diff = jpeg_category(bits, True)
                dc_value = prev_dc[channel_idx] + dc_diff
                prev_dc[channel_idx] = dc_value
                rle_values = []
                coeff_idx = 0
                while coeff_idx < 63:
                    symbol = huffman_decode_ac(ac_tree)
                    if symbol is None:
                        break
                    run, category = symbol
                    if category > 0:
                        bits = read_bits(category)
                        if bits is None:
                            break
                        value = jpeg_category(bits, True)
                    else:
                        value = 0
                    coeff_idx += run
                    rle_values.append((run, value))
                    if run == 0 and category == 0:
                        break
                    coeff_idx += 1
                if not rle_values: break
                ac_coeffs = jpeg_rle_decoding(rle_values)
                zigzag = [dc_value] + ac_coeffs + [0] * (63 - len(ac_coeffs))
                block = jpeg_zigzag(zigzag, True)
                for i in range(8):
                    for j in range(8):
                        block[i][j] *= qtable[i][j]
                block = jpeg_dct8x8(block, True)
                for i in range(8):
                    for j in range(8):
                        val = int(round(block[i][j] + 128))
                        block[i][j] = max(0, min(255, val))
                channels[channel_idx].append(block)
    y_matrix = [[0] * width for _ in range(height)]
    cb_matrix = [[0] * width for _ in range(height)]
    cr_matrix = [[0] * width for _ in range(height)]
    for c, matrix in enumerate([y_matrix, cb_matrix, cr_matrix]):
        for block_idx, block in enumerate(channels[c]):
            row = (block_idx // blocks_per_row) * 8
            col = (block_idx % blocks_per_row) * 8
            for i in range(8):
                for j in range(8):
                    if row + i < height and col + j < width:
                        matrix[row + i][col + j] = block[i][j]
    rgb_matrix = []
    for i in range(height):
        row = []
        for j in range(width):
            y, cb, cr = y_matrix[i][j], cb_matrix[i][j], cr_matrix[i][j]
            r, g, b = ycbcr2rgb([y, cb, cr])
            row.append((r, g, b))
        rgb_matrix.append(row)
    return rgb_matrix


class JPEG(BaseImage):
    def read(self, filename: str) -> None:
        print("⚠⚠⚠ There seems to be a problem with the implementation of this method. ⚠⚠⚠")
        lqtable = None
        cqtable = None
        with open(filename, "rb") as rb:
            if rb.read(2) != b"\xff\xd8":
                raise ValueError("Not a valid JPEG file: missing SOI marker (FFD8)")
            while True:
                while rb.read(1) != b"\xff":
                    continue
                marker_code = rb.read(1)
                if not marker_code:
                    raise ValueError("Unexpected end of file after marker start byte (FF)")
                marker_code = marker_code[0]
                if marker_code == 0xDA:
                    segment_length = int.from_bytes(rb.read(2), "big")
                    rb.read(segment_length - 2)
                    break
                segment_length = int.from_bytes(rb.read(2), "big")
                segment_data = rb.read(segment_length - 2)
                if marker_code == 0xC0:
                    self._height = int.from_bytes(segment_data[1:3], "big")
                    self._width = int.from_bytes(segment_data[3:5], "big")
                elif marker_code == 0xDB:
                    i = 0
                    while i < len(segment_data):
                        info_byte = segment_data[i]
                        precision = (info_byte >> 4) & 0x0F
                        table_id = info_byte & 0x0F
                        i += 1
                        if precision != 0:
                            raise NotImplementedError("16-bit quantization tables are not supported")
                        table_data = jpeg_zigzag(segment_data[i:i + 64], True)
                        i += 64
                        if table_id == 0:
                            lqtable = table_data
                        elif table_id == 1:
                            cqtable = table_data
            if lqtable is None or cqtable is None:
                raise ValueError("JPEG file is missing required quantization tables")
            scan_data = rb.read()
            if scan_data:
                pixels = jpeg_decode_pixels(scan_data, lqtable, cqtable, self._width, self._height)

                self._pixels = [bytearray(b"".join(map(bytes, row))) for row in pixels]
            else:
                raise ValueError("No image data found in the JPEG file")

    def write(self, filename: str = None, quality=50) -> bytes:
        if self._pixels is None:
            raise ValueError("Please create or read an image first")
        pixel_matrix = []
        bpp = _CHANNEL[self._color_mode]
        for row in self._pixels:
            pixel_row = []
            for i in range(0, len(row), bpp):
                pixel_row.append(row[i: i + bpp])
            pixel_matrix.append(pixel_row)
        encoded, lqtable, cqtable = jpeg_encode_pixels(pixel_matrix, quality)
        image = b"".join([
            b"\xff\xd8",
            b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00",
            b"\xff\xdb\x00C\x00",
            bytes(jpeg_zigzag(lqtable)),
            b"\xff\xdb\x00C\x01",
            bytes(jpeg_zigzag(cqtable)),
            b"\xff\xc0\x00\x11\x08",
            self._width.to_bytes(2, "big"), self._height.to_bytes(2, "big"),
            b"\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01",
            bytes.fromhex("FFC4001F0000010501010101010100000000000000000102030405060708090A0B"),
            bytes.fromhex("FFC400B5100002010303020403050504040000017D01020300041105122131410613516107227114328191A10823"
                          "42B1C11552D1F02433627282090A161718191A25262728292A3435363738393A434445464748494A535455565758"
                          "595A636465666768696A737475767778797A838485868788898A92939495969798999AA2A3A4A5A6A7A8A9AAB2B3"
                          "B4B5B6B7B8B9BAC2C3C4C5C6C7C8C9CAD2D3D4D5D6D7D8D9DAE1E2E3E4E5E6E7E8E9EAF1F2F3F4F5F6F7F8F9FA"),
            bytes.fromhex("FFC4001F0100030101010101010101010000000000000102030405060708090A0B"),
            bytes.fromhex("FFC400B51100020102040403040705040400010277000102031104052131061241510761711322328108144291A1"
                          "B1C109233352F0156272D10A162434E125F11718191A262728292A35363738393A434445464748494A5354555657"
                          "58595A636465666768696A737475767778797A82838485868788898A92939495969798999AA2A3A4A5A6A7A8A9AA"
                          "B2B3B4B5B6B7B8B9BAC2C3C4C5C6C7C8C9CAD2D3D4D5D6D7D8D9DAE2E3E4E5E6E7E8E9EAF2F3F4F5F6F7F8F9FA"),
            b"\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00",
            encoded,
            b"\xff\xd9"
        ])
        if filename:
            with open(filename, "wb") as wb:
                wb.write(image)
        else:
            return image
