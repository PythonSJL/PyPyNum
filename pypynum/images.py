import collections
import heapq
import math
import warnings
import zlib
from .kernels import matmul8x8kernel
from .types import Any, arr

_CHANNEL = {'GRAY': 1, 'RGB': 3, 'INDEX': 1, 'GRAYA': 2, 'RGBA': 4}
_DEPTH = {1: ('INDEX', 1), 2: ('INDEX', 2), 4: ('INDEX', 4), 8: ('INDEX', 8), 24: ('RGB', 8), 32: ('RGBA', 8),
          48: ('RGB', 16), 64: ('RGBA', 16)}
_MODE = {'GRAY': 0, 'RGB': 2, 'INDEX': 3, 'GRAYA': 4, 'RGBA': 6}
_NUMBER = {0: 'GRAY', 2: 'RGB', 3: 'INDEX', 4: 'GRAYA', 6: 'RGBA'}
_ADAM7_PASSES = ((0, 0, 8, 8), (0, 4, 8, 8), (4, 0, 8, 4), (0, 2, 4, 4), (2, 0, 4, 2), (0, 1, 2, 2), (1, 0, 2, 1))
_STD_LUMA_QTABLE = [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]]
_STD_CHROMA_QTABLE = [[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                      [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]]
_DHT_DC0 = bytes.fromhex('FFC4001F0000010501010101010100000000000000000102030405060708090A0B')
_DHT_AC0 = bytes.fromhex(
    'FFC400B5100002010303020403050504040000017D01020300041105122131410613516107227114328191A1082342B1C11552D1F02433627282090A161718191A25262728292A3435363738393A434445464748494A535455565758595A636465666768696A737475767778797A838485868788898A92939495969798999AA2A3A4A5A6A7A8A9AAB2B3B4B5B6B7B8B9BAC2C3C4C5C6C7C8C9CAD2D3D4D5D6D7D8D9DAE1E2E3E4E5E6E7E8E9EAF1F2F3F4F5F6F7F8F9FA')
_DHT_DC1 = bytes.fromhex('FFC4001F0100030101010101010101010000000000000102030405060708090A0B')
_DHT_AC1 = bytes.fromhex(
    'FFC400B51100020102040403040705040400010277000102031104052131061241510761711322328108144291A1B1C109233352F0156272D10A162434E125F11718191A262728292A35363738393A434445464748494A535455565758595A636465666768696A737475767778797A82838485868788898A92939495969798999AA2A3A4A5A6A7A8A9AAB2B3B4B5B6B7B8B9BAC2C3C4C5C6C7C8C9CAD2D3D4D5D6D7D8D9DAE2E3E4E5E6E7E8E9EAF2F3F4F5F6F7F8F9FA')


def __dht2dict(dht_bytes):
    length = int.from_bytes(dht_bytes[2:4], 'big')
    counts = dht_bytes[5:21]
    values = dht_bytes[21:2 + length]
    huff_dict = {}
    code = 0
    val_idx = 0
    for bit_len in range(1, 17):
        for _ in range(counts[bit_len - 1]):
            huff_dict[values[val_idx] >> 4, values[val_idx] & 15] = format(code, '0{}b'.format(bit_len))
            val_idx += 1
            code += 1
        code <<= 1
    return huff_dict


__AC = (__dht2dict(_DHT_AC0), __dht2dict(_DHT_AC1))


def rgb2ycbcr(weights: arr) -> tuple:
    r, g, b, *_ = weights
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168735892 * r - 0.331264108 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418687589 * g - 0.081312411 * b + 128
    return (round(y), round(cb), round(cr))


def ycbcr2rgb(weights: arr) -> tuple:
    y, cb, cr, *_ = weights
    cb = cb - 128
    cr = cr - 128
    r = y + 1.402 * cr
    g = y - 0.344136286 * cb - 0.714136286 * cr
    b = y + 1.772 * cb
    return (round(r), round(g), round(b))


def _find_closest_color(color, palette, octree_root=None) -> int:
    r, g, b = color[:3]
    if octree_root is not None:
        return octree_root.find_index_greedy(r, g, b)
    min_dist = float('inf')
    best_index = 0
    for i, pc in enumerate(palette):
        pr, pg, pb = pc[:3]
        dist = ((r - pr) * 0.3) ** 2 + ((g - pg) * 0.59) ** 2 + ((b - pb) * 0.11) ** 2
        if dist < min_dist:
            min_dist = dist
            best_index = i
        if min_dist == 0:
            break
    return best_index


class BaseImage:

    def __init__(self) -> None:
        self._bit_depth = None
        self._color_mode = None
        self._width = None
        self._height = None
        self._pixels = None
        self._palette = None

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

    @property
    def palette(self) -> list:
        return self._palette

    @palette.setter
    def palette(self, value: list) -> None:
        if value is not None:
            for color in value:
                if not (isinstance(color, tuple) and len(color) in (3, 4) and all(
                        (isinstance(c, int) and 0 <= c <= 255 for c in color))):
                    raise ValueError('Palette colors must be RGB(A) tuples with values 0-255')
        self._palette = value

    def info(self) -> dict:
        return {'width': self._width, 'height': self._height, 'bit_depth': self._bit_depth,
                'color_mode': self._color_mode}

    def new(self, width: int, height: int, color: tuple = (), color_mode: str = 'RGB', bit_depth: int = 8,
            palette: list = None) -> None:
        color_mode = color_mode.strip().upper()
        supported = ('RGB', 'RGBA', 'INDEX', 'GRAY', 'GRAYA')
        if color_mode not in supported:
            raise ValueError(
                'The current version supports only one color mode from {}'.format(', '.join(map(repr, supported))))
        if color_mode == 'INDEX':
            if bit_depth not in (1, 2, 4, 8):
                raise ValueError('Indexed color mode only supports 1, 2, 4, or 8-bit depth')
            max_palette_size = 1 << bit_depth
            if palette is None or len(palette) == 0:
                palette = self._generate_default_palette(max_palette_size)
            if len(palette) > max_palette_size:
                raise ValueError(
                    'For {}-bit depth, palette size cannot exceed {} colors'.format(bit_depth, max_palette_size))
            self._palette = palette
        elif color_mode == 'GRAY':
            if bit_depth not in (1, 2, 4, 8, 16):
                raise ValueError('GRAY color mode only supports 1, 2, 4, 8, or 16-bit depth')
            self._palette = None
        elif color_mode == 'GRAYA':
            if bit_depth not in (8, 16):
                raise ValueError('GRAYA color mode only supports 8 or 16-bit depth')
            self._palette = None
        else:
            if bit_depth not in (8, 16):
                raise ValueError('The bit depth of a single channel must be 8 or 16')
            self._palette = None
        self._width = width
        self._height = height
        if color_mode == 'INDEX':
            if color:
                if len(color) == 1 and isinstance(color[0], int):
                    if not 0 <= color[0] < len(self._palette):
                        raise ValueError('Color index must be 0-{}'.format(len(self._palette) - 1))
                    pixel_bytes = color[0].to_bytes(1, 'big')
                else:
                    closest_idx = _find_closest_color(color, self._palette)
                    pixel_bytes = closest_idx.to_bytes(1, 'big')
            else:
                pixel_bytes = b'\x00'
            self._pixels = [[pixel_bytes for _ in range(width)] for _ in range(height)]
        elif color_mode == 'GRAY':
            channel_count = _CHANNEL['GRAY']
            if bit_depth < 8:
                max_val = (1 << bit_depth) - 1
                if color:
                    if len(color) != 1 or not (isinstance(color[0], int) and 0 <= color[0] <= max_val):
                        raise ValueError('GRAY color must be a single integer between 0 and {}'.format(max_val))
                    pixel_bytes = color[0].to_bytes(1, 'big')
                else:
                    pixel_bytes = b'\x00'
                self._pixels = [[pixel_bytes for _ in range(width)] for _ in range(height)]
            else:
                maximum = (1 << bit_depth) - 1
                if color:
                    color = tuple(color)
                else:
                    color = (maximum,)
                if len(color) != channel_count:
                    raise ValueError('The specified color can only be a {} tuple'.format(color_mode))
                if not all([isinstance(_, int) and 0 <= _ <= maximum for _ in color]):
                    raise ValueError(
                        'The color value must be an integer between 0 and {} when the bit depth is {}'.format(maximum,
                                                                                                              bit_depth))
                bytes_per_value = bit_depth // 8
                pixel_bytes = b''.join([c.to_bytes(bytes_per_value, 'big') for c in color])
                self._pixels = [[pixel_bytes for _ in range(width)] for _ in range(height)]
        else:
            channel_count = _CHANNEL[color_mode]
            maximum = (1 << bit_depth) - 1
            if color:
                color = tuple(color)
            else:
                color = (maximum,) * channel_count
            if len(color) != channel_count:
                raise ValueError(
                    'The specified color can only be an {}-channel tuple for {}'.format(channel_count, color_mode))
            if not all([isinstance(_, int) and 0 <= _ <= maximum for _ in color]):
                raise ValueError(
                    'The color value must be an integer between 0 and {} when the bit depth is {}'.format(maximum,
                                                                                                          bit_depth))
            bytes_per_value = bit_depth // 8
            pixel_bytes = b''.join([c.to_bytes(bytes_per_value, 'big') for c in color])
            self._pixels = [[pixel_bytes for _ in range(width)] for _ in range(height)]
        self._bit_depth = bit_depth
        self._color_mode = color_mode

    @staticmethod
    def _generate_default_palette(num_colors: int) -> list:
        palette = []
        if num_colors == 2:
            palette = [(0, 0, 0), (255, 255, 255)]
        elif num_colors == 4:
            palette = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 255)]
        elif num_colors <= 16:
            palette = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                       (192, 192, 192), (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255),
                       (255, 0, 255), (0, 255, 255), (255, 255, 255)]
            palette = palette[:num_colors]
        else:
            for r in range(6):
                for g in range(6):
                    for b in range(6):
                        palette.append((r * 51, g * 51, b * 51))
            remaining = num_colors - len(palette)
            for i in range(remaining):
                v = round(i * 255 / max(remaining - 1, 1)) if remaining > 1 else 0
                palette.append((v, v, v))
            palette = palette[:num_colors]
        return palette

    def __getitem__(self, item: tuple) -> tuple:
        x, y = item
        pixel_data = self._pixels[y][x]
        if self._color_mode == 'INDEX':
            index = pixel_data[0]
            return (index,)
        elif self._color_mode == 'GRAY' and self._bit_depth < 8:
            return (pixel_data[0],)
        else:
            channel_count = _CHANNEL[self._color_mode]
            bytes_per_value = self._bit_depth // 8
            pixel_size = channel_count * bytes_per_value
            if self._bit_depth == 8:
                return tuple(pixel_data)
            else:
                return tuple([int.from_bytes(pixel_data[i:i + bytes_per_value], 'big') for i in
                              range(0, pixel_size, bytes_per_value)])

    def __setitem__(self, key: tuple, value: tuple) -> None:
        x, y = key
        color = tuple(value)
        if self._color_mode == 'INDEX':
            if len(color) == 1 and isinstance(color[0], int):
                idx = color[0]
                if self._palette is None or not 0 <= idx < len(self._palette):
                    raise ValueError('Color index must be an integer between 0 and {}'.format(
                        len(self._palette) - 1 if self._palette else 0))
                self._pixels[y][x] = idx.to_bytes(1, 'big')
            else:
                if self._palette is None:
                    raise ValueError('No palette available for color matching')
                idx = _find_closest_color(color, self._palette)
                self._pixels[y][x] = idx.to_bytes(1, 'big')
        elif self._color_mode == 'GRAY' and self._bit_depth < 8:
            if len(color) != 1 or not isinstance(color[0], int):
                raise ValueError('GRAY pixel must be a single integer')
            max_val = (1 << self._bit_depth) - 1
            if not 0 <= color[0] <= max_val:
                raise ValueError(
                    'GRAY value must be between 0 and {} for {}-bit depth'.format(max_val, self._bit_depth))
            self._pixels[y][x] = color[0].to_bytes(1, 'big')
        else:
            channel_count = _CHANNEL[self._color_mode]
            if len(color) != channel_count:
                raise ValueError('The specified color can only be an {}-channel tuple for {}'.format(channel_count,
                                                                                                     self._color_mode))
            maximum = (1 << self._bit_depth) - 1
            if not all([isinstance(_, int) and 0 <= _ <= maximum for _ in color]):
                raise ValueError(
                    'The color value must be an integer between 0 and {} when the bit depth is {}'.format(maximum,
                                                                                                          self._bit_depth))
            bytes_per_value = self._bit_depth // 8
            self._pixels[y][x] = b''.join([val.to_bytes(bytes_per_value, 'big') for val in value])

    def __repr__(self) -> str:
        return self.__class__.__name__ + str(sorted(self.info().items())).replace("',", ':').replace("'", '')

    def _get_8bit_rgb_matrix(self) -> list:
        pixels_8bit = []
        source_max = (1 << self.bit_depth) - 1
        is_scale = source_max != 255 and source_max > 0
        bytes_per_val = self.bit_depth // 8
        if self.color_mode == 'INDEX':
            for row in self._pixels:
                row_8bit = []
                for p in row:
                    idx = p[0]
                    if self._palette and idx < len(self._palette):
                        row_8bit.append(self._palette[idx][:3])
                    else:
                        row_8bit.append((0, 0, 0))
                pixels_8bit.append(row_8bit)
        elif self.color_mode in ('GRAY', 'GRAYA'):
            for row in self._pixels:
                row_8bit = []
                for p in row:
                    if self._bit_depth < 8:
                        gray_val = p[0]
                        max_val = (1 << self._bit_depth) - 1
                        if max_val > 0:
                            gray_val = round(gray_val * 255 / max_val)
                    elif bytes_per_val == 1:
                        gray_val = p[0]
                    else:
                        gray_val = int.from_bytes(p, 'big')
                        if is_scale and self._bit_depth >= 8:
                            gray_val = round(gray_val * 255 / source_max)
                    row_8bit.append((gray_val, gray_val, gray_val))
                pixels_8bit.append(row_8bit)
        elif self.color_mode in ('RGB', 'RGBA'):
            for row in self._pixels:
                row_8bit = []
                for p in row:
                    if bytes_per_val == 1:
                        r, g, b = (p[0], p[1], p[2])
                    else:
                        r = int.from_bytes(p[0:bytes_per_val], 'big')
                        g = int.from_bytes(p[bytes_per_val:2 * bytes_per_val], 'big')
                        b = int.from_bytes(p[2 * bytes_per_val:3 * bytes_per_val], 'big')
                    if is_scale:
                        r = round(r * 255 / source_max)
                        g = round(g * 255 / source_max)
                        b = round(b * 255 / source_max)
                    row_8bit.append((r, g, b))
                pixels_8bit.append(row_8bit)
        else:
            pixels_8bit = [[(0, 0, 0) for _ in range(self._width)] for _ in range(self._height)]
        return pixels_8bit

    def _generate_palette(self, max_colors=256, method='octree') -> tuple:
        if self._pixels is None:
            raise ValueError('Please create or read an image first')
        pixels_matrix = self._get_8bit_rgb_matrix()
        pixels = [pixel for row in pixels_matrix for pixel in row]
        if not pixels:
            empty_palette = [(0, 0, 0)] * max_colors
            return (empty_palette, None)
        if method == 'octree':
            return octree_quantize(pixels, max_colors)
        else:
            raise ValueError("Unsupported quantization method. Choose 'octree'")

    def convert(self, format_name: str, color_mode: str = None, bit_depth: int = None) -> 'BaseImage':
        if self._pixels is None:
            raise ValueError('Please create or read an image first before converting')
        format_name = format_name.strip().upper()
        format_map = {'BMP': BMP, 'PNG': PNG, 'JPEG': JPEG, 'GIF': GIF}
        if format_name not in format_map:
            raise ValueError(
                'Unsupported target format: {}. Supported formats are: {}'.format(format_name, ', '.join(format_map)))
        img_class = format_map[format_name]
        new_img = img_class()
        color_mode = color_mode.strip().upper() if color_mode is not None else self.color_mode
        bit_depth = bit_depth if bit_depth is not None else self.bit_depth
    
        if format_name == 'BMP':
            if color_mode not in ('RGB', 'RGBA', 'INDEX'):
                color_mode = 'RGB'
            if bit_depth > 8:
                bit_depth = 8
        elif format_name == 'JPEG':
            if color_mode == 'GRAYA':
                color_mode = 'GRAY'
            elif color_mode != 'GRAY':
                color_mode = 'RGB'
            bit_depth = 8
        elif format_name == 'GIF':
            if color_mode != 'INDEX':
                color_mode = 'INDEX'
            if bit_depth not in (1, 2, 4, 8):
                bit_depth = 8
    
        if color_mode == 'INDEX':
            if bit_depth not in (1, 2, 4, 8):
                bit_depth = 8
            max_colors = 1 << bit_depth
            pixels_matrix = self._get_8bit_rgb_matrix()
            pixels_flat = [p for row in pixels_matrix for p in row]
            palette, octree_root = octree_quantize(pixels_flat, max_colors)
            new_img.new(self.width, self.height, color_mode='INDEX', bit_depth=bit_depth, palette=palette)
            for y in range(self.height):
                for x in range(self.width):
                    index = _find_closest_color(pixels_matrix[y][x], palette, octree_root)
                    new_img[x, y] = (index,)
            return new_img
    
        if color_mode == 'GRAY':
            if bit_depth not in (1, 2, 4, 8, 16):
                bit_depth = 8
        elif color_mode == 'GRAYA':
            if bit_depth not in (8, 16):
                bit_depth = 8
        elif bit_depth not in (8, 16):
            bit_depth = 8
    
        new_img.new(self.width, self.height, color_mode=color_mode, bit_depth=bit_depth)
        target_max = (1 << bit_depth) - 1
    
        for y in range(self.height):
            for x in range(self.width):
                source_pixel = self[x, y]
                source_max = (1 << self.bit_depth) - 1
    
                # ========== 第一步：将源像素统一转为 8-bit RGBA（保留完整色彩） ==========
                if self.color_mode == 'INDEX':
                    idx = source_pixel[0]
                    if self._palette and idx < len(self._palette):
                        pal_color = self._palette[idx]
                    else:
                        pal_color = (0, 0, 0)
                    r, g, b = pal_color[:3]
                    alpha = pal_color[3] if len(pal_color) > 3 else 255
                elif self.color_mode == 'GRAY':
                    g_val = source_pixel[0]
                    if source_max != 255 and source_max > 0:
                        g_val = round(g_val * 255 / source_max)
                    r, g, b, alpha = (g_val, g_val, g_val, 255)
                elif self.color_mode == 'GRAYA':
                    g_val, a_val = (source_pixel[0], source_pixel[1])
                    if source_max != 255 and source_max > 0:
                        g_val = round(g_val * 255 / source_max)
                        a_val = round(a_val * 255 / source_max)
                    r, g, b, alpha = (g_val, g_val, g_val, a_val)
                elif self.color_mode == 'RGB':
                    r, g, b = source_pixel
                    if source_max != 255 and source_max > 0:
                        r = round(r * 255 / source_max)
                        g = round(g * 255 / source_max)
                        b = round(b * 255 / source_max)
                    alpha = 255
                elif self.color_mode == 'RGBA':
                    r, g, b, a = source_pixel
                    if source_max != 255 and source_max > 0:
                        r = round(r * 255 / source_max)
                        g = round(g * 255 / source_max)
                        b = round(b * 255 / source_max)
                        a = round(a * 255 / source_max)
                    alpha = a
                else:
                    r, g, b, alpha = (0, 0, 0, 255)
    
                # ========== 第二步：从 8-bit RGBA 转为目标色彩模式 ==========
                def _scale(val, max_val):
                    return round(val * max_val / 255) if max_val != 255 else val
    
                if color_mode == 'GRAY':
                    gray = round(0.299 * r + 0.587 * g + 0.114 * b)
                    target_pixel = (_scale(gray, target_max),)
                elif color_mode == 'GRAYA':
                    gray = round(0.299 * r + 0.587 * g + 0.114 * b)
                    target_pixel = (_scale(gray, target_max), _scale(alpha, target_max))
                elif color_mode == 'RGB':
                    target_pixel = (_scale(r, target_max), _scale(g, target_max), _scale(b, target_max))
                elif color_mode == 'RGBA':
                    target_pixel = (_scale(r, target_max), _scale(g, target_max), _scale(b, target_max), _scale(alpha, target_max))
                else:
                    target_pixel = (0,)
    
                new_img[x, y] = target_pixel
    
        return new_img


class BMP(BaseImage):

    def new(self, width: int, height: int, color: tuple = (), color_mode: str = 'RGB', bit_depth: int = 8,
            palette: list = None) -> None:
        if bit_depth > 8:
            raise NotImplementedError('The bit_depth is too high for this implementation')
        super().new(width, height, color, color_mode, bit_depth, palette)

    @staticmethod
    def rgb2bgr(color: tuple) -> tuple:
        r, g, b, *a = color
        return (b, g, r, a[0]) if a else (b, g, r)

    @staticmethod
    def bgr2rgb(color: tuple) -> tuple:
        b, g, r, *a = color
        return (r, g, b, a[0]) if a else (r, g, b)

    def read(self, filename: str) -> None:
        with open(filename, 'rb') as rb:
            if rb.read(2) != b'BM':
                raise ValueError('The file is not a valid BMP image')
            rb.read(12)
            header_size = int.from_bytes(rb.read(4), 'little')
            if header_size != 40:
                raise ValueError('The BMP image has an invalid header size of {} bytes'.format(header_size))
            width, height = (int.from_bytes(rb.read(4), 'little'), int.from_bytes(rb.read(4), 'little'))
            planes = int.from_bytes(rb.read(2), 'little')
            bits_per_pixel = int.from_bytes(rb.read(2), 'little')
            compression = int.from_bytes(rb.read(4), 'little')
            rb.read(20)
            if planes != 1:
                raise ValueError(
                    'The BMP image must have a single plane of color data, but found {} planes'.format(planes))
            if compression != 0:
                raise ValueError(
                    'The BMP image must not be compressed, but found compression type {}'.format(compression))
            if bits_per_pixel not in _DEPTH:
                raise ValueError('The BMP image has an unsupported color depth of {} bits'.format(bits_per_pixel))
            self._color_mode, self._bit_depth = _DEPTH[bits_per_pixel]
            palette_size = 1 << bits_per_pixel if bits_per_pixel <= 8 else 0
            if palette_size:
                palette_data = rb.read(palette_size * 4)
                self._palette = []
                for i in range(palette_size):
                    b = palette_data[i * 4]
                    g = palette_data[i * 4 + 1]
                    r = palette_data[i * 4 + 2]
                    self._palette.append((r, g, b))
            else:
                self._palette = None
            if bits_per_pixel <= 8:
                self._color_mode = 'INDEX'
                row_size = (width * bits_per_pixel + 31 & -32) >> 3
                pixels = rb.read(row_size * height)
                self._pixels = []
                for row in range(height - 1, -1, -1):
                    start = row * row_size
                    row_data = pixels[start:start + row_size]
                    row_pixels = []
                    for byte_idx in range(width):
                        if bits_per_pixel == 8:
                            index = row_data[byte_idx]
                        elif bits_per_pixel == 4:
                            byte = row_data[byte_idx // 2]
                            index = byte >> 4 if byte_idx % 2 == 0 else byte & 15
                        elif bits_per_pixel == 2:
                            byte = row_data[byte_idx // 4]
                            shift = 6 - 2 * (byte_idx % 4)
                            index = byte >> shift & 3
                        elif bits_per_pixel == 1:
                            byte = row_data[byte_idx // 8]
                            index = byte >> 7 - byte_idx % 8 & 1
                        row_pixels.append(index.to_bytes(1, 'big'))
                    self._pixels.append(row_pixels)
                self._width, self._height = (width, height)
            else:
                length = _CHANNEL[self._color_mode]
                row_size = (width * bits_per_pixel + 31 & -32) >> 3
                pixels = rb.read(row_size * height)
                self._pixels = []
                for row in range(height - 1, -1, -1):
                    start = row * row_size
                    raw_row = pixels[start:start + width * length]
                    self._pixels.append(
                        [bytes(self.bgr2rgb(raw_row[i:i + length])) for i in range(0, width * length, length)])
                self._width, self._height = (width, height)

    def write(self, filename: str = None) -> bytes:
        if self._pixels is None:
            raise ValueError('Please create or read an image first')
        if self._color_mode in ('GRAY', 'GRAYA'):
            target_mode = 'RGBA' if self._color_mode == 'GRAYA' else 'RGB'
            temp_img = BMP()
            temp_img.new(self._width, self._height, color_mode=target_mode, bit_depth=8)
            for y in range(self._height):
                for x in range(self._width):
                    pixel = self[x, y]
                    if self._color_mode == 'GRAY':
                        g = pixel[0]
                        source_max = (1 << self._bit_depth) - 1
                        if source_max != 255 and source_max > 0:
                            g = round(g * 255 / source_max)
                        temp_img[x, y] = (g, g, g)
                    else:
                        g, a = (pixel[0], pixel[1])
                        source_max = (1 << self._bit_depth) - 1
                        if source_max != 255 and source_max > 0:
                            g = round(g * 255 / source_max)
                            a = round(a * 255 / source_max)
                        temp_img[x, y] = (g, g, g, a)
            return temp_img.write(filename)
        if self._color_mode == 'INDEX':
            if self._palette is None:
                raise ValueError('Indexed color mode requires a palette')
            bits_per_pixel = self._bit_depth
            max_colors = 1 << bits_per_pixel
            palette_data = b''
            for i in range(max_colors):
                if i < len(self._palette):
                    r, g, b = self._palette[i][:3]
                    palette_data += bytes([b, g, r, 0])
                else:
                    palette_data += b'\x00\x00\x00\x00'
            packed_rows = []
            for row in self._pixels:
                indices = [p[0] for p in row]
                packed_byte_list = bytearray()
                if bits_per_pixel == 8:
                    packed_byte_list.extend(indices)
                elif bits_per_pixel == 4:
                    for i in range(0, len(indices), 2):
                        byte_val = indices[i] << 4
                        if i + 1 < len(indices):
                            byte_val |= indices[i + 1]
                        packed_byte_list.append(byte_val)
                elif bits_per_pixel == 2:
                    for i in range(0, len(indices), 4):
                        byte_val = indices[i] << 6
                        if i + 1 < len(indices):
                            byte_val |= indices[i + 1] << 4
                        if i + 2 < len(indices):
                            byte_val |= indices[i + 2] << 2
                        if i + 3 < len(indices):
                            byte_val |= indices[i + 3]
                        packed_byte_list.append(byte_val)
                elif bits_per_pixel == 1:
                    for i in range(0, len(indices), 8):
                        byte_val = 0
                        for j in range(8):
                            if i + j < len(indices):
                                byte_val |= indices[i + j] << 7 - j
                        packed_byte_list.append(byte_val)
                row_bytes = bytes(packed_byte_list)
                padding_len = (4 - len(row_bytes) % 4) % 4
                packed_rows.append(row_bytes + b'\x00' * padding_len)
            pixels = b''.join(reversed(packed_rows))
            image = b''.join([b'BM', (len(pixels) + len(palette_data) + 54).to_bytes(4, 'little'), b'\x00' * 4,
                              (len(palette_data) + 54).to_bytes(4, 'little'), b'(\x00\x00\x00',
                              self._width.to_bytes(4, 'little'), self._height.to_bytes(4, 'little'), b'\x01\x00',
                              bits_per_pixel.to_bytes(2, 'little'), b'\x00' * 24, palette_data, pixels])
        else:
            channel_size = _CHANNEL[self._color_mode]
            filled_width = (self._width * self._bit_depth * channel_size + 31 & -32) >> 3
            filler = b'\x00' * (filled_width - self._width * channel_size)
            pixel_data_list = []
            for row in self._pixels:
                bgr_row = [bytes(self.rgb2bgr(pixel)) for pixel in row]
                bgr_row.append(filler)
                pixel_data_list.append(b''.join(bgr_row))
            pixels = b''.join(reversed(pixel_data_list))
            pixel_size = filled_width * self._height * channel_size
            palette_size = 1 << self._bit_depth if self._color_mode == 'INDEX' else 0
            image = b''.join([b'BM', (pixel_size + palette_size + 54).to_bytes(4, 'little'), b'\x00' * 4,
                              (palette_size + 54).to_bytes(4, 'little'), b'(\x00\x00\x00',
                              self._width.to_bytes(4, 'little'), self._height.to_bytes(4, 'little'), b'\x01\x00',
                              (channel_size * self._bit_depth).to_bytes(2, 'little'), b'\x00' * 24, pixels])
        if filename:
            with open(filename, 'wb') as wb:
                wb.write(image)
        return image


class PNG(BaseImage):

    def read(self, filename: str) -> None:
        with open(filename, 'rb') as rb:
            part = rb.read(8)
            if part != b'\x89PNG\r\n\x1a\n':
                raise ValueError('The file is not a valid PNG image')
            temp = []
            part = rb.read(4)
            length = int.from_bytes(part, 'big')
            part = rb.read(4)
            temp.append(part)
            if part != b'IHDR':
                raise ValueError('Missing IHDR chunk')
            if length != 13:
                raise ValueError('The length of IHDR chunk must be 13')
            width, height = (rb.read(4), rb.read(4))
            self._width, self._height = (int.from_bytes(width, 'big'), int.from_bytes(height, 'big'))
            part = rb.read(5)
            temp.append(width)
            temp.append(height)
            temp.append(part)
            self._bit_depth = part[0]
            self._color_mode = _NUMBER[part[1]]
            interlace = part[4]
            part = rb.read(4)
            if zlib.crc32(b''.join(temp)) != int.from_bytes(part, 'big'):
                warnings.warn('IHDR chunk verification failed', RuntimeWarning)
            plte_data = None
            trns_data = None
            pixels = []
            while True:
                part = rb.read(4)
                length = int.from_bytes(part, 'big')
                chunk_type = rb.read(4)
                if chunk_type == b'PLTE':
                    plte_data = rb.read(length)
                    rb.read(4)
                elif chunk_type == b'tRNS':
                    trns_data = rb.read(length)
                    rb.read(4)
                elif chunk_type == b'IDAT':
                    temp = rb.read(length)
                    pixels.append(temp)
                    part = rb.read(4)
                    if zlib.crc32(b'IDAT' + temp) != int.from_bytes(part, 'big'):
                        warnings.warn('IDAT chunk verification failed for chunk number {}'.format(len(pixels)),
                                      RuntimeWarning)
                elif chunk_type == b'IEND':
                    part = rb.read(4)
                    if part != b'\xaeB`\x82':
                        warnings.warn('IEND chunk verification failed', RuntimeWarning)
                    break
                else:
                    rb.read(length + 4)
            if self._color_mode == 'INDEX':
                if plte_data is None:
                    raise ValueError('Indexed PNG must have PLTE chunk')
                self._palette = []
                for i in range(0, len(plte_data), 3):
                    r, g, b = (plte_data[i], plte_data[i + 1], plte_data[i + 2])
                    self._palette.append((r, g, b))
                if trns_data:
                    for i, alpha in enumerate(trns_data):
                        if i < len(self._palette):
                            r, g, b = self._palette[i]
                            self._palette[i] = (r, g, b, alpha)
            else:
                self._palette = None
            if self._color_mode == 'GRAY' and trns_data:
                if len(trns_data) >= 2:
                    trans_gray = int.from_bytes(trns_data[:2], 'big')
                    self._gray_trns_value = trans_gray
                else:
                    self._gray_trns_value = None
            de = zlib.decompressobj().decompress(b''.join(pixels))
            if interlace == 0:
                length = len(de)
                line = length // self._height
                if self._color_mode == 'INDEX' or (self._color_mode == 'GRAY' and self._bit_depth < 8):
                    filtered_rows = []
                    prev_row = None
                    for y in range(0, length, line):
                        filter_type = de[y]
                        row_data = de[y + 1:y + line]
                        filtered_row = [[b] for b in row_data]
                        original_row = png_reverse_filter(filtered_row, prev_row, filter_type)
                        if self._bit_depth == 8:
                            unpacked_row = [bytes(pixel) for pixel in original_row]
                        else:
                            unpacked_row = []
                            for pixel in original_row:
                                byte_val = pixel[0]
                                if self._bit_depth == 4:
                                    unpacked_row.append(bytes([byte_val >> 4 & 15]))
                                    unpacked_row.append(bytes([byte_val & 15]))
                                elif self._bit_depth == 2:
                                    unpacked_row.append(bytes([byte_val >> 6 & 3]))
                                    unpacked_row.append(bytes([byte_val >> 4 & 3]))
                                    unpacked_row.append(bytes([byte_val >> 2 & 3]))
                                    unpacked_row.append(bytes([byte_val & 3]))
                                elif self._bit_depth == 1:
                                    for bit in range(7, -1, -1):
                                        unpacked_row.append(bytes([byte_val >> bit & 1]))
                            unpacked_row = unpacked_row[:self._width]
                        filtered_rows.append(unpacked_row)
                        prev_row = original_row
                    self._pixels = [[bytes(pixel) for pixel in row] for row in filtered_rows]
                elif self._color_mode in ('GRAY', 'GRAYA', 'RGB', 'RGBA'):
                    step = _CHANNEL[self._color_mode] * (self._bit_depth // 8)
                    filtered_rows = []
                    prev_row = None
                    for y in range(0, length, line):
                        filter_type = de[y]
                        row_data = de[y + 1:y + line]
                        filtered_row = [row_data[x:x + step] for x in range(0, len(row_data), step)]
                        original_row = png_reverse_filter(filtered_row, prev_row, filter_type)
                        filtered_rows.append(original_row)
                        prev_row = original_row
                    self._pixels = [[bytes(pixel) for pixel in row] for row in filtered_rows]
            else:
                if self._color_mode == 'INDEX' or (self._color_mode == 'GRAY' and self._bit_depth < 8):
                    default_pixel = b'\x00'
                else:
                    default_pixel = b'\x00' * (_CHANNEL[self._color_mode] * (self._bit_depth // 8))
                self._pixels = [[default_pixel for _ in range(self._width)] for _ in range(self._height)]
                offset = 0
                for start_row, start_col, row_step, col_step in _ADAM7_PASSES:
                    if start_row >= self._height or start_col >= self._width:
                        continue
                    pass_height = (self._height - start_row + row_step - 1) // row_step
                    pass_width = (self._width - start_col + col_step - 1) // col_step
                    if pass_height == 0 or pass_width == 0:
                        continue
                    if self._color_mode == 'INDEX' or (self._color_mode == 'GRAY' and self._bit_depth < 8):
                        row_data_bytes = (pass_width * self._bit_depth + 7) // 8
                    else:
                        row_data_bytes = pass_width * _CHANNEL[self._color_mode] * (self._bit_depth // 8)
                    prev_row = None
                    for pass_y in range(pass_height):
                        filter_type = de[offset]
                        offset += 1
                        row_data = de[offset:offset + row_data_bytes]
                        offset += row_data_bytes
                        if self._color_mode == 'INDEX' or (self._color_mode == 'GRAY' and self._bit_depth < 8):
                            filtered_row = [[b] for b in row_data]
                        else:
                            step = _CHANNEL[self._color_mode] * (self._bit_depth // 8)
                            filtered_row = [row_data[x:x + step] for x in range(0, len(row_data), step)]
                        original_row = png_reverse_filter(filtered_row, prev_row, filter_type)
                        prev_row = original_row
                        if (self._color_mode == 'INDEX' or self._color_mode == 'GRAY') and self._bit_depth < 8:
                            unpacked = []
                            for pixel in original_row:
                                byte_val = pixel[0]
                                if self._bit_depth == 4:
                                    unpacked.append(bytes([byte_val >> 4 & 15]))
                                    unpacked.append(bytes([byte_val & 15]))
                                elif self._bit_depth == 2:
                                    unpacked.append(bytes([byte_val >> 6 & 3]))
                                    unpacked.append(bytes([byte_val >> 4 & 3]))
                                    unpacked.append(bytes([byte_val >> 2 & 3]))
                                    unpacked.append(bytes([byte_val & 3]))
                                elif self._bit_depth == 1:
                                    for bit in range(7, -1, -1):
                                        unpacked.append(bytes([byte_val >> bit & 1]))
                            pixel_list = unpacked[:pass_width]
                        else:
                            pixel_list = [bytes(pixel) for pixel in original_row]
                        actual_y = start_row + pass_y * row_step
                        for pass_x, pixel in enumerate(pixel_list):
                            actual_x = start_col + pass_x * col_step
                            self._pixels[actual_y][actual_x] = pixel
            if hasattr(self, '_gray_trns_value') and self._gray_trns_value is not None:
                trans_val = self._gray_trns_value
                source_max = (1 << self._bit_depth) - 1
                new_pixels = []
                for row in self._pixels:
                    new_row = []
                    for pixel_bytes in row:
                        if self._bit_depth <= 8:
                            gray_val = pixel_bytes[0]
                            if self._bit_depth < 8:
                                mapped_val = round(
                                    gray_val * source_max / ((1 << self._bit_depth) - 1)) if source_max > 0 else 0
                            else:
                                mapped_val = gray_val
                            alpha = 0 if mapped_val == trans_val else 255
                        else:
                            gray_val = int.from_bytes(pixel_bytes, 'big')
                            alpha = 0 if gray_val == trans_val else 255
                        new_row.append(bytes([gray_val if self._bit_depth <= 8 else 0, alpha]))
                    new_pixels.append(new_row)
                self._pixels = new_pixels
                self._color_mode = 'GRAYA'
                self._bit_depth = 8
                del self._gray_trns_value

    def write(self, filename: str = None, apply_filter: str = 'none', interlace: bool = False) -> bytes:
        if self._pixels is None:
            raise ValueError('Please create or read an image first')
        ihdr = b''.join(
            [b'IHDR', self._width.to_bytes(4, 'big'), self._height.to_bytes(4, 'big'), bytes([self._bit_depth]),
             bytes([_MODE[self._color_mode]]), b'\x00\x00', b'\x01' if interlace else b'\x00'])
        filter_map = ('none', 'sub', 'up', 'average', 'paeth', 'adaptive')
        apply_filter = apply_filter.strip().lower()
        if apply_filter not in filter_map:
            raise ValueError('apply_filter must be one of ' + str(filter_map))
        if not interlace:
            pixel_matrix = []
            for row in self._pixels:
                if (self._color_mode == 'INDEX' or self._color_mode == 'GRAY') and self._bit_depth < 8:
                    packed_row = []
                    byte_val = 0
                    bits_in_byte = 0
                    for pixel_bytes in row:
                        idx = pixel_bytes[0]
                        bits_needed = self._bit_depth
                        byte_val = byte_val << bits_needed | idx
                        bits_in_byte += bits_needed
                        if bits_in_byte == 8:
                            packed_row.append([byte_val])
                            byte_val = 0
                            bits_in_byte = 0
                    if bits_in_byte > 0:
                        byte_val <<= 8 - bits_in_byte
                        packed_row.append([byte_val])
                    pixel_matrix.append(packed_row)
                else:
                    pixel_row = []
                    for pixel_bytes in row:
                        pixel_row.append(list(pixel_bytes))
                    pixel_matrix.append(pixel_row)
            precomputed = []
            num_rows = len(pixel_matrix)
            for r in range(num_rows):
                row_filters_data = []
                current_row = pixel_matrix[r]
                prev_row = pixel_matrix[r - 1] if r > 0 else None
                for f_type in range(5):
                    filtered_pixels = png_apply_filter(current_row, prev_row, f_type)
                    row_bytes = b''.join(map(bytes, filtered_pixels))
                    row_filters_data.append(row_bytes)
                precomputed.append(row_filters_data)
            candidates = []
            if apply_filter == 'adaptive':
                mixed_rows = []
                for r in range(num_rows):
                    if r == 0:
                        best_f = 1
                    else:
                        best_f = 0
                        best_ent = float('inf')
                        for f in (0, 1, 2, 3, 4):
                            ent = entropy(precomputed[r][f])
                            if ent < best_ent:
                                best_ent = ent
                                best_f = f
                    mixed_rows.append(bytes([best_f]) + precomputed[r][best_f])
                candidates.append(mixed_rows)
                first_f = 1
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
            min_size = float('inf')
            for scanlines in candidates:
                c_data = zlib.compress(b''.join(scanlines))
                size = len(c_data)
                if size < min_size:
                    min_size = size
                    best = c_data
            idat = b'IDAT' + best
        else:
            all_scanlines = []
            for start_row, start_col, row_step, col_step in _ADAM7_PASSES:
                if start_row >= self._height or start_col >= self._width:
                    continue
                pass_height = (self._height - start_row + row_step - 1) // row_step
                pass_width = (self._width - start_col + col_step - 1) // col_step
                if pass_height == 0 or pass_width == 0:
                    continue
                pass_pixel_matrix = []
                for pass_y in range(pass_height):
                    actual_y = start_row + pass_y * row_step
                    if (self._color_mode == 'INDEX' or self._color_mode == 'GRAY') and self._bit_depth < 8:
                        indices = [self._pixels[actual_y][start_col + px * col_step][0] for px in range(pass_width)]
                        packed_row = []
                        byte_val = 0
                        bits_in_byte = 0
                        for idx in indices:
                            byte_val = byte_val << self._bit_depth | idx
                            bits_in_byte += self._bit_depth
                            if bits_in_byte == 8:
                                packed_row.append([byte_val])
                                byte_val = 0
                                bits_in_byte = 0
                        if bits_in_byte > 0:
                            byte_val <<= 8 - bits_in_byte
                            packed_row.append([byte_val])
                        pass_pixel_matrix.append(packed_row)
                    else:
                        pixel_row = [list(self._pixels[actual_y][start_col + px * col_step]) for px in
                                     range(pass_width)]
                        pass_pixel_matrix.append(pixel_row)
                pass_precomputed = []
                for r in range(len(pass_pixel_matrix)):
                    row_filters_data = []
                    current_row = pass_pixel_matrix[r]
                    prev_row = pass_pixel_matrix[r - 1] if r > 0 else None
                    for f_type in range(5):
                        filtered_pixels = png_apply_filter(current_row, prev_row, f_type)
                        row_bytes = b''.join(map(bytes, filtered_pixels))
                        row_filters_data.append(row_bytes)
                    pass_precomputed.append(row_filters_data)
                for r in range(len(pass_precomputed)):
                    if apply_filter == 'adaptive':
                        if r == 0:
                            best_f = 1
                        else:
                            best_f = 0
                            best_ent = float('inf')
                            for f in (0, 1, 2, 3, 4):
                                ent = entropy(pass_precomputed[r][f])
                                if ent < best_ent:
                                    best_ent = ent
                                    best_f = f
                        all_scanlines.append(bytes([best_f]) + pass_precomputed[r][best_f])
                    else:
                        fixed_f = filter_map.index(apply_filter)
                        all_scanlines.append(bytes([fixed_f]) + pass_precomputed[r][fixed_f])
            idat = b'IDAT' + zlib.compress(b''.join(all_scanlines))
        if self._color_mode == 'INDEX':
            if self._palette is None:
                raise ValueError('Indexed color mode requires a palette')
            plte_data = b''
            trns_data = b''
            has_alpha = any((len(c) == 4 for c in self._palette))
            for color in self._palette:
                if len(color) >= 3:
                    plte_data += bytes(color[:3])
                if has_alpha:
                    trns_data += bytes([color[3] if len(color) > 3 else 255])
            plte_chunk = b'PLTE' + plte_data
            plte_crc = zlib.crc32(plte_chunk).to_bytes(4, 'big')
            plte_length = len(plte_data).to_bytes(4, 'big')
            trns_chunk = b''
            if trns_data:
                trns_chunk_data = b'tRNS' + trns_data
                trns_crc = zlib.crc32(trns_chunk_data).to_bytes(4, 'big')
                trns_length = len(trns_data).to_bytes(4, 'big')
                trns_chunk = trns_length + trns_chunk_data + trns_crc
            image = b''.join(
                [b'\x89PNG\r\n\x1a\n', b'\x00\x00\x00\r', ihdr, zlib.crc32(ihdr).to_bytes(4, 'big'), plte_length,
                 plte_chunk, plte_crc, trns_chunk, b'\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9',
                 b'\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05',
                 b'\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d',
                 (len(idat) - 4).to_bytes(4, 'big'), idat, zlib.crc32(idat).to_bytes(4, 'big'),
                 b'\x00\x00\x00\x00IEND\xaeB`\x82'])
        elif self._color_mode == 'GRAY':
            trns_chunk = b''
            image = b''.join([b'\x89PNG\r\n\x1a\n', b'\x00\x00\x00\r', ihdr, zlib.crc32(ihdr).to_bytes(4, 'big'),
                              b'\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9',
                              b'\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05',
                              b'\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d', trns_chunk,
                              (len(idat) - 4).to_bytes(4, 'big'), idat, zlib.crc32(idat).to_bytes(4, 'big'),
                              b'\x00\x00\x00\x00IEND\xaeB`\x82'])
        else:
            image = b''.join([b'\x89PNG\r\n\x1a\n', b'\x00\x00\x00\r', ihdr, zlib.crc32(ihdr).to_bytes(4, 'big'),
                              b'\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9',
                              b'\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05',
                              b'\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d',
                              (len(idat) - 4).to_bytes(4, 'big'), idat, zlib.crc32(idat).to_bytes(4, 'big'),
                              b'\x00\x00\x00\x00IEND\xaeB`\x82'])
        if filename:
            with open(filename, 'wb') as wb:
                wb.write(image)
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
                filtered_val = current_val - left_val & 255
            elif filter_type == 2:
                above_val = above_pixels[i][c] if above_pixels and i < len(above_pixels) else 0
                filtered_val = current_val - above_val & 255
            elif filter_type == 3:
                left_val = pixels[i - 1][c] if i > 0 else 0
                above_val = above_pixels[i][c] if above_pixels and i < len(above_pixels) else 0
                filtered_val = current_val - (left_val + above_val) // 2 & 255
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
                filtered_val = current_val - predictor & 255
            else:
                raise ValueError('Unknown filter type')
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
                original_val = filtered_val + left_val & 255
            elif filter_type == 2:
                above_val = above_pixels[i][c] if above_pixels and i < len(above_pixels) else 0
                original_val = filtered_val + above_val & 255
            elif filter_type == 3:
                left_val = original_pixels[i - 1][c] if i > 0 else 0
                above_val = above_pixels[i][c] if above_pixels and i < len(above_pixels) else 0
                original_val = filtered_val + (left_val + above_val) // 2 & 255
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
                original_val = filtered_val + predictor & 255
            else:
                raise ValueError('Unknown filter type')
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
        dct8x8, dct8x8t = (dct8x8t, dct8x8)
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
        raise TypeError('All elements in the sequence must be integers')
    encoded = []
    i = 0
    length = len(sequence)
    pos = 0  # 记录当前已确认的系数总位置数
    
    while i != length:
        count = 0
        while i != length and count != 15 and (sequence[i] == 0):
            count += 1
            i += 1
        if i != length:
            item = sequence[i]
            i += 1
            encoded.append((count, item))
            pos += count + 1  # 跳过 count 个零，加上 1 个非零值
            
    # 移除尾部无效的 ZRL (15, 0)
    while encoded and encoded[-1] == (15, 0):
        del encoded[-1]
        pos -= 16  # 同步减去 ZRL 代表的 16 个系数
        
    # 如果未填满序列长度，必须追加 EOB (0, 0)
    if pos < length:
        encoded.append((0, 0))
        
    return encoded

def jpeg_rle_decoding(sequence: arr) -> list:
    if set(map(len, sequence)) != {2}:
        raise TypeError('All elements in the sequence must be tuples of length 2')
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
        raise ValueError('Quality must be between 0 and 100')
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
    target_rows = (rows + block_size - 1) // block_size * block_size
    target_cols = (cols + block_size - 1) // block_size * block_size
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
            blocks.append([row_block[k][j:j + block_size] for k in range(block_size)])
    return blocks


def jpeg_luma_dc_huff(data: Any, reverse: bool = False) -> Any:
    if reverse:
        fixed_codes = {'00': 0, '010': 1, '011': 2, '100': 3, '101': 4}
        if data in fixed_codes:
            return fixed_codes[data]
        data = list(data)
        endswith = data.pop()
        if endswith == '0' and len(data) >= 2:
            if set(data) == {'1'}:
                num_ones = len(data)
                return num_ones + 3
        return None
    if data >= 5:
        return '1' * (data - 3) + '0'
    else:
        return ('00', '010', '011', '100', '101')[data]


def jpeg_chroma_dc_huff(data: Any, reverse: bool = False) -> Any:
    if reverse:
        if data == '00':
            return 0
        if data == '01':
            return 1
        data = list(data)
        endswith = data.pop()
        if endswith == '0' and len(data) >= 1:
            if set(data) == {'1'}:
                num_ones = len(data)
                return num_ones + 1
        return None
    if data == 0:
        return '00'
    elif data == 1:
        return '01'
    else:
        return '1' * (data - 1) + '0'


def jpeg_category(data: Any, reverse: bool = False) -> Any:
    if reverse:
        if data == '':
            return 0
        length = len(data)
        value = int(data, 2)
        if data[0] == '0':
            value -= (1 << length) - 1
        return value
    if data == 0:
        return (0, '')
    length = data.bit_length()
    if data < 0:
        data += (1 << length) - 1
    return (length, '{:0{}b}'.format(data, length))


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
    return (
        [''.join((dc(length), value, ''.join([ac[a, b] + c for a, b, c in sublist]))) for (length, value), sublist in
         zip(map(jpeg_category, dpcm_values), encoded)], qtable)


def jpeg_encode_pixels(pixels: arr, quality: int, subsampling: str = '4:4:4') -> tuple:
    is_gray = len(pixels[0][0]) == 1
    subsampling_map = {'4:4:4': (1, 1), '4:2:2': (2, 1), '4:2:0': (2, 2), '4:1:1': (4, 1), '4:4:0': (1, 2), '4:1:0': (4, 2)}
    if isinstance(subsampling, str):
        yh, yv = subsampling_map.get(subsampling, (1, 1))
    else:
        yh, yv = subsampling
    if is_gray:
        y_matrix = [[p[0] if isinstance(p, (bytes, tuple, list)) else p for p in row] for row in pixels]
        yh, yv = (1, 1)
    else:
        split = tuple(map(list, zip(*[map(list, zip(*map(rgb2ycbcr, row))) for row in pixels])))
        y_matrix, cb_matrix, cr_matrix = (split[0], split[1], split[2])
    height = len(y_matrix)
    width = len(y_matrix[0])
    target_h = (height + yv * 8 - 1) // (yv * 8) * yv * 8
    target_w = (width + yh * 8 - 1) // (yh * 8) * yh * 8

    def pad_matrix(matrix, t_h, t_w):
        h = len(matrix)
        w = len(matrix[0])
        padded_matrix = [row[:] for row in matrix]
        if w < t_w:
            for r in padded_matrix:
                r.extend([r[-1]] * (t_w - w))
        if h < t_h:
            last_row = padded_matrix[-1][:]
            padded_matrix.extend([last_row[:] for _ in range(t_h - h)])
        return padded_matrix

    y_matrix_padded = pad_matrix(y_matrix, target_h, target_w)

    def split_into_blocks(matrix):
        h = len(matrix)
        w = len(matrix[0])
        blocks = []
        for r in range(0, h, 8):
            row_blocks = []
            for c in range(0, w, 8):
                block = [[matrix[r + i][c + j] for j in range(8)] for i in range(8)]
                row_blocks.append(block)
            blocks.append(row_blocks)
        return blocks

    y_blocks_2d = split_into_blocks(y_matrix_padded)
    cb_blocks_2d = cr_blocks_2d = None
    if not is_gray:

        def downsample_and_split(matrix, h_factor, v_factor):
            orig_h = len(matrix)
            orig_w = len(matrix[0])
            ds_h = (orig_h + v_factor - 1) // v_factor
            ds_w = (orig_w + h_factor - 1) // h_factor
            ds_matrix = [[0] * ds_w for _ in range(ds_h)]
            for r in range(ds_h):
                for c in range(ds_w):
                    val = 0.0
                    weight = 0.0
                    for vr in range(v_factor):
                        for hc in range(h_factor):
                            orig_r = r * v_factor + vr
                            orig_c = c * h_factor + hc
                            if orig_r < orig_h and orig_c < orig_w:
                                val += matrix[orig_r][orig_c]
                                weight += 1.0
                    ds_matrix[r][c] = round(val / weight) if weight > 0 else 0
            block_h = (ds_h + 7) // 8 * 8
            block_w = (ds_w + 7) // 8 * 8
            padded_ds = [row[:] for row in ds_matrix]
            if len(padded_ds[0]) < block_w:
                for row in padded_ds:
                    row.extend([row[-1]] * (block_w - len(row)))
            if len(padded_ds) < block_h:
                last_row = padded_ds[-1][:]
                padded_ds.extend([last_row[:] for _ in range(block_h - len(padded_ds))])
            return split_into_blocks(padded_ds)

        cb_blocks_2d = downsample_and_split(cb_matrix, yh, yv)
        cr_blocks_2d = downsample_and_split(cr_matrix, yh, yv)
    lqtable = jpeg_adjust_qtable(_STD_LUMA_QTABLE, quality)
    cqtable = jpeg_adjust_qtable(_STD_CHROMA_QTABLE, quality)
    prev_dc = [0, 0, 0]

    def process_block(block, qtable):
        for i in range(8):
            for j in range(8):
                block[i][j] -= 128
        block = jpeg_dct8x8(block)
        for i in range(8):
            for j in range(8):
                block[i][j] = round(block[i][j] / qtable[i][j])
        return jpeg_zigzag(block)

    def encode_block(zigzag, mode):
        nonlocal prev_dc
        current_dc = zigzag.pop(0)
        dpcm = current_dc - prev_dc[mode]
        prev_dc[mode] = current_dc
        rle = jpeg_rle_encoding(zigzag)
        encoded_rle = [(x, *jpeg_category(y)) for x, y in rle]
        dc_coder = jpeg_chroma_dc_huff if mode else jpeg_luma_dc_huff
        ac_coder = __AC[0 if mode == 0 else 1]
        dc_len, dc_val = jpeg_category(dpcm)
        dc_bits = dc_coder(dc_len)
        ac_bits = ''.join([ac_coder[a, b] + c for a, b, c in encoded_rle])
        return dc_bits + dc_val + ac_bits

    mcu_rows = target_h // (yv * 8)
    mcu_cols = target_w // (yh * 8)
    bin_str = ''
    for r_mcu in range(mcu_rows):
        for c_mcu in range(mcu_cols):
            for vr in range(yv):
                for hc in range(yh):
                    r_b = r_mcu * yv + vr
                    c_b = c_mcu * yh + hc
                    block = y_blocks_2d[r_b][c_b]
                    zigzag = process_block([row[:] for row in block], lqtable)
                    bin_str += encode_block(zigzag, 0)
            if not is_gray:
                block = cb_blocks_2d[r_mcu][c_mcu]
                zigzag = process_block([row[:] for row in block], cqtable)
                bin_str += encode_block(zigzag, 1)
                block = cr_blocks_2d[r_mcu][c_mcu]
                zigzag = process_block([row[:] for row in block], cqtable)
                bin_str += encode_block(zigzag, 2)
    bit_length = len(bin_str)
    if bit_length % 8 != 0:
        bin_str += '1' * (8 - bit_length % 8)
    data = int(bin_str, 2).to_bytes(len(bin_str) // 8, 'big').replace(b'\xff', b'\xff\x00')
    return (data, lqtable, cqtable if not is_gray else lqtable, yh, yv, 1 if is_gray else 3)


def jpeg_decode_pixels(scan_data: bytes, q_tables: dict, huff_tables: dict, width: int, height: int, yh: int = 1,
                       yv: int = 1, comp_ids: list = None, comp_info: dict = None, comp_huff: dict = None,
                       rst_offsets: list = None) -> list:
    if not scan_data:
        return []
    bin_str = format(int.from_bytes(scan_data, 'big'), '0{}b'.format(len(scan_data) * 8))
    bit_ptr = 0
    rst_bit_offsets = [off * 8 for off in rst_offsets or []]
    rst_idx = 0

    def read_bits(n):
        nonlocal bit_ptr
        if bit_ptr + n > len(bin_str):
            return None
        bits = bin_str[bit_ptr:bit_ptr + n]
        bit_ptr += n
        return bits

    def huffman_decode(tree):
        nonlocal bit_ptr
        current_node = tree
        while True:
            if bit_ptr >= len(bin_str):
                return None
            bit = bin_str[bit_ptr]
            bit_ptr += 1
            current_node = current_node[int(bit)]
            if current_node is None or not isinstance(current_node, list):
                return None
            if isinstance(current_node[0], int):
                return current_node

    prev_dc = {c_id: 0 for c_id in comp_ids or [0]}

    def decode_one_block(comp_id):
        nonlocal prev_dc
        q_id = comp_info[comp_id]['q_id']
        qtable = q_tables.get(q_id, q_tables.get(0))
        dc_tree_id = comp_huff[comp_id]['dc']
        ac_tree_id = comp_huff[comp_id]['ac']
        dc_tree = huff_tables['dc'].get(dc_tree_id)
        ac_tree = huff_tables['ac'].get(ac_tree_id)
        dc_symbol = huffman_decode(dc_tree)
        if dc_symbol is None:
            return [[0] * 8 for _ in range(8)]
        dc_category = dc_symbol[0]
        dc_diff = 0
        if dc_category > 0:
            bits = read_bits(dc_category)
            if bits is None:
                return [[0] * 8 for _ in range(8)]
            dc_diff = jpeg_category(bits, True)
        dc_value = prev_dc[comp_id] + dc_diff
        prev_dc[comp_id] = dc_value
        rle_values = []
        coeff_idx = 0
        while coeff_idx < 63:
            symbol = huffman_decode(ac_tree)
            if symbol is None:
                break
            run, category = (symbol[0], symbol[1])
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
        return block

    y_matrix = [[0] * width for _ in range(height)]
    is_gray = len(comp_ids) == 1
    cb_height = (height + yv - 1) // yv
    cb_width = (width + yh - 1) // yh
    cb_matrix = [[0] * cb_width for _ in range(cb_height)]
    cr_matrix = [[0] * cb_width for _ in range(cb_height)]
    mcu_w = yh * 8
    mcu_h = yv * 8
    mcu_cols = (width + mcu_w - 1) // mcu_w
    mcu_rows = (height + mcu_h - 1) // mcu_h
    for r_mcu in range(mcu_rows):
        for c_mcu in range(mcu_cols):
            if rst_idx < len(rst_bit_offsets) and bit_ptr >= rst_bit_offsets[rst_idx]:
                bit_ptr = (bit_ptr + 7) // 8 * 8
                prev_dc = {c_id: 0 for c_id in comp_ids}
                rst_idx += 1
            y_comp_id = comp_ids[0]
            for vr in range(yv):
                for hc in range(yh):
                    block = decode_one_block(y_comp_id)
                    r_start = r_mcu * mcu_h + vr * 8
                    c_start = c_mcu * mcu_w + hc * 8
                    for i in range(8):
                        for j in range(8):
                            if r_start + i < height and c_start + j < width:
                                y_matrix[r_start + i][c_start + j] = block[i][j]
            if not is_gray:
                cb_comp_id = comp_ids[1]
                cb_block = decode_one_block(cb_comp_id)
                r_start_cb = r_mcu * 8
                c_start_cb = c_mcu * 8
                for i in range(8):
                    for j in range(8):
                        r = r_start_cb + i
                        c = c_start_cb + j
                        if r < cb_height and c < cb_width:
                            cb_matrix[r][c] = cb_block[i][j]
                cr_comp_id = comp_ids[2]
                cr_block = decode_one_block(cr_comp_id)
                for i in range(8):
                    for j in range(8):
                        r = r_start_cb + i
                        c = c_start_cb + j
                        if r < cb_height and c < cb_width:
                            cr_matrix[r][c] = cr_block[i][j]
    if is_gray:
        return [[(c,) for c in row] for row in y_matrix]
    else:
        final_pixels = []
        for r in range(height):
            row_pixels = []
            for c in range(width):
                y_val = y_matrix[r][c]
                cr_pos = (r + 0.5) * cb_height / height - 0.5
                cc_pos = (c + 0.5) * cb_width / width - 0.5
                cr_pos = max(0.0, min(cb_height - 1.0, cr_pos))
                cc_pos = max(0.0, min(cb_width - 1.0, cc_pos))
                r0 = int(cr_pos)
                c0 = int(cc_pos)
                r1 = min(r0 + 1, cb_height - 1)
                c1 = min(c0 + 1, cb_width - 1)
                fr = cr_pos - r0
                fc = cc_pos - c0
                cb_val = cb_matrix[r0][c0] * (1 - fr) * (1 - fc) + cb_matrix[r0][c1] * (1 - fr) * fc + cb_matrix[r1][
                    c0] * fr * (1 - fc) + cb_matrix[r1][c1] * fr * fc
                cr_val = cr_matrix[r0][c0] * (1 - fr) * (1 - fc) + cr_matrix[r0][c1] * (1 - fr) * fc + cr_matrix[r1][
                    c0] * fr * (1 - fc) + cr_matrix[r1][c1] * fr * fc
                rgb = ycbcr2rgb((y_val, cb_val, cr_val))
                row_pixels.append(tuple((max(0, min(255, int(v))) for v in rgb)))
            final_pixels.append(row_pixels)
        return final_pixels


def jpeg_decode_progressive_pixels(scans_info, q_tables, huff_tables, width, height, yh, yv, comp_ids, comp_info, comp_huff):
    """
    解码渐进式 JPEG 像素数据（支持多扫描段）。
    修复了 AC 首次扫描 ZRL 的判断逻辑。
    修复了非交织扫描段按光栅顺序解码，解决色度子采样下的方块错位问题。
    """
    is_gray = len(comp_ids) == 1
    mcu_w = yh * 8
    mcu_h = yv * 8
    mcu_cols = (width + mcu_w - 1) // mcu_w
    mcu_rows = (height + mcu_h - 1) // mcu_h

    y_brows = mcu_rows * yv
    y_bcols = mcu_cols * yh
    y_coeffs = [[0] * 64 for _ in range(y_brows * y_bcols)]
    cb_coeffs = cr_coeffs = None
    if not is_gray:
        cb_coeffs = [[0] * 64 for _ in range(mcu_rows * mcu_cols)]
        cr_coeffs = [[0] * 64 for _ in range(mcu_rows * mcu_cols)]

    prev_dc = {cid: 0 for cid in comp_ids}

    for scan in scans_info:
        scan_data = scan['data']
        scan_comp_ids = scan['comp_ids']
        Ss = scan['Ss']
        Se = scan['Se']
        Ah = scan['Ah']
        Al = scan['Al']
        rst_offsets = scan.get('rst_offsets', [])

        bin_str = format(int.from_bytes(scan_data, 'big'), '0{}b'.format(len(scan_data) * 8))
        bit_ptr = 0
        rst_bit_offsets = [off * 8 for off in rst_offsets]
        rst_idx = 0

        for cid in scan_comp_ids:
            prev_dc[cid] = 0

        def _read_bits(n):
            nonlocal bit_ptr
            if bit_ptr + n > len(bin_str):
                return None
            bits = bin_str[bit_ptr:bit_ptr + n]
            bit_ptr += n
            return bits

        def _huffman_decode(tree):
            nonlocal bit_ptr
            if tree is None:
                return None
            current_node = tree
            while True:
                if bit_ptr >= len(bin_str):
                    return None
                bit = bin_str[bit_ptr]
                bit_ptr += 1
                current_node = current_node[int(bit)]
                if current_node is None:
                    return None
                if not isinstance(current_node, list):
                    return None
                if len(current_node) >= 2 and isinstance(current_node[0], int):
                    return current_node

        def _check_rst():
            nonlocal bit_ptr, rst_idx
            if rst_idx < len(rst_bit_offsets) and bit_ptr >= rst_bit_offsets[rst_idx]:
                bit_ptr = (bit_ptr + 7) // 8 * 8
                for cid in scan_comp_ids:
                    prev_dc[cid] = 0
                rst_idx += 1

        def _decode_one_progressive_block(comp_id, block, dc_tree, ac_tree):
            if Ss == 0 and Se == 0:
                # ========== DC 扫描段 ==========
                if Ah == 0:
                    dc_symbol = _huffman_decode(dc_tree)
                    if dc_symbol is None:
                        return
                    dc_category = dc_symbol[0]
                    dc_diff = 0
                    if dc_category > 0:
                        bits = _read_bits(dc_category)
                        if bits is None:
                            return
                        dc_diff = jpeg_category(bits, True)
                    dc_value = prev_dc[comp_id] + dc_diff
                    prev_dc[comp_id] = dc_value
                    block[0] = dc_value << Al
                else:
                    bits = _read_bits(1)
                    if bits is None:
                        return
                    if int(bits):
                        if block[0] >= 0:
                            block[0] += (1 << Al)
                        else:
                            block[0] -= (1 << Al)
            else:
                # ========== AC 扫描段 ==========
                if Ah == 0:
                    coeff_idx = Ss
                    while coeff_idx <= Se:
                        symbol = _huffman_decode(ac_tree)
                        if symbol is None:
                            break
                        run, category = symbol[0], symbol[1]
                        if run == 0 and category == 0:  # EOB
                            break
                        if run == 15 and category == 0: # ZRL (16个零游程)
                            coeff_idx += 16
                            continue
                        coeff_idx += run
                        if coeff_idx > Se:
                            break
                        if category > 0:
                            bits = _read_bits(category)
                            if bits is None:
                                break
                            value = jpeg_category(bits, True)
                            block[coeff_idx] = value << Al
                            coeff_idx += 1
                else:
                    # AC 精细扫描
                    coeff_idx = Ss
                    while coeff_idx <= Se:
                        if block[coeff_idx] != 0:
                            bits = _read_bits(1)
                            if bits is not None and int(bits):
                                if block[coeff_idx] > 0:
                                    block[coeff_idx] += (1 << Al)
                                else:
                                    block[coeff_idx] -= (1 << Al)
                            coeff_idx += 1
                        else:
                            symbol = _huffman_decode(ac_tree)
                            if symbol is None:
                                break
                            run, category = symbol[0], symbol[1]
                            if run == 0 and category == 0:  # EOB
                                coeff_idx += 1
                                while coeff_idx <= Se:
                                    if block[coeff_idx] != 0:
                                        bits = _read_bits(1)
                                        if bits is not None and int(bits):
                                            if block[coeff_idx] > 0:
                                                block[coeff_idx] += (1 << Al)
                                            else:
                                                block[coeff_idx] -= (1 << Al)
                                    coeff_idx += 1
                                break
                            while run > 0 and coeff_idx <= Se:
                                if block[coeff_idx] != 0:
                                    bits = _read_bits(1)
                                    if bits is not None and int(bits):
                                        if block[coeff_idx] > 0:
                                            block[coeff_idx] += (1 << Al)
                                        else:
                                            block[coeff_idx] -= (1 << Al)
                                else:
                                    run -= 1
                                coeff_idx += 1
                            if coeff_idx > Se:
                                break
                            if category == 1:
                                bits = _read_bits(1)
                                if bits is not None:
                                    sign = 1 if int(bits) else -1
                                    block[coeff_idx] = sign << Al
                                coeff_idx += 1
                            elif category > 1:
                                bits = _read_bits(category)
                                if bits is not None:
                                    value = jpeg_category(bits, True)
                                    block[coeff_idx] = value << Al
                                coeff_idx += 1

        is_interleaved = len(scan_comp_ids) > 1

        if is_interleaved:
            # === 交织扫描段（如 DC 扫描段）：按 MCU 顺序 ===
            for r_mcu in range(mcu_rows):
                for c_mcu in range(mcu_cols):
                    _check_rst()
                    for comp_id in scan_comp_ids:
                        info = comp_info[comp_id]
                        v_samp = info['v']
                        h_samp = info['h']
                        dc_tree = huff_tables['dc'].get(comp_huff[comp_id]['dc'])
                        ac_tree = huff_tables['ac'].get(comp_huff[comp_id]['ac'])

                        for vr in range(v_samp):
                            for hc in range(h_samp):
                                if comp_id == comp_ids[0]:
                                    br = r_mcu * yv + vr
                                    bc = c_mcu * yh + hc
                                    block = y_coeffs[br * y_bcols + bc]
                                elif comp_id == comp_ids[1]:
                                    block = cb_coeffs[r_mcu * mcu_cols + c_mcu]
                                else:
                                    block = cr_coeffs[r_mcu * mcu_cols + c_mcu]
                                
                                _decode_one_progressive_block(comp_id, block, dc_tree, ac_tree)
        else:
            # === 非交织扫描段（如 Y AC、Cb AC、Cr AC）：按光栅顺序 ===
            comp_id = scan_comp_ids[0]
            dc_tree = huff_tables['dc'].get(comp_huff[comp_id]['dc'])
            ac_tree = huff_tables['ac'].get(comp_huff[comp_id]['ac'])

            if comp_id == comp_ids[0]:  # Y 分量
                for br in range(y_brows):
                    for bc in range(y_bcols):
                        _check_rst()
                        block = y_coeffs[br * y_bcols + bc]
                        _decode_one_progressive_block(comp_id, block, dc_tree, ac_tree)
            elif comp_id == comp_ids[1]:  # Cb 分量
                for br in range(mcu_rows):
                    for bc in range(mcu_cols):
                        _check_rst()
                        block = cb_coeffs[br * mcu_cols + bc]
                        _decode_one_progressive_block(comp_id, block, dc_tree, ac_tree)
            else:  # Cr 分量
                for br in range(mcu_rows):
                    for bc in range(mcu_cols):
                        _check_rst()
                        block = cr_coeffs[br * mcu_cols + bc]
                        _decode_one_progressive_block(comp_id, block, dc_tree, ac_tree)

    # ===================== 将系数转换为像素 =====================
    y_matrix = [[0] * width for _ in range(height)]
    if not is_gray:
        cb_height = (height + yv - 1) // yv
        cb_width = (width + yh - 1) // yh
        cb_matrix = [[0] * cb_width for _ in range(cb_height)]
        cr_matrix = [[0] * cb_width for _ in range(cb_height)]

    y_qtable = q_tables.get(comp_info[comp_ids[0]]['q_id'], q_tables.get(0))
    for br in range(y_brows):
        for bc in range(y_bcols):
            block_zz = y_coeffs[br * y_bcols + bc][:]
            block_natural = jpeg_zigzag(block_zz, True)
            for i in range(8):
                for j in range(8):
                    block_natural[i][j] *= y_qtable[i][j]
            block_natural = jpeg_dct8x8(block_natural, True)
            for i in range(8):
                for j in range(8):
                    py = br * 8 + i
                    px = bc * 8 + j
                    if py < height and px < width:
                        val = int(round(block_natural[i][j] + 128))
                        y_matrix[py][px] = max(0, min(255, val))

    if not is_gray:
        for comp_idx in range(2):
            coeffs = cb_coeffs if comp_idx == 0 else cr_coeffs
            cid = comp_ids[comp_idx + 1]
            qtable = q_tables.get(comp_info[cid]['q_id'], q_tables.get(0))
            target = cb_matrix if comp_idx == 0 else cr_matrix
            for br in range(mcu_rows):
                for bc in range(mcu_cols):
                    block_zz = coeffs[br * mcu_cols + bc][:]
                    block_natural = jpeg_zigzag(block_zz, True)
                    for i in range(8):
                        for j in range(8):
                            block_natural[i][j] *= qtable[i][j]
                    block_natural = jpeg_dct8x8(block_natural, True)
                    for i in range(8):
                        for j in range(8):
                            py = br * 8 + i
                            px = bc * 8 + j
                            if py < len(target) and px < len(target[0]):
                                val = int(round(block_natural[i][j] + 128))
                                target[py][px] = max(0, min(255, val))

    if is_gray:
        return [[(c,) for c in row] for row in y_matrix]

    final_pixels = []
    for r in range(height):
        row_pixels = []
        for c in range(width):
            y_val = y_matrix[r][c]
            cr_pos = (r + 0.5) * cb_height / height - 0.5
            cc_pos = (c + 0.5) * cb_width / width - 0.5
            cr_pos = max(0.0, min(cb_height - 1.0, cr_pos))
            cc_pos = max(0.0, min(cb_width - 1.0, cc_pos))
            r0 = int(cr_pos)
            c0 = int(cc_pos)
            r1 = min(r0 + 1, cb_height - 1)
            c1 = min(c0 + 1, cb_width - 1)
            fr = cr_pos - r0
            fc = cc_pos - c0
            cb_val = cb_matrix[r0][c0] * (1 - fr) * (1 - fc) + cb_matrix[r0][c1] * (1 - fr) * fc + \
                     cb_matrix[r1][c0] * fr * (1 - fc) + cb_matrix[r1][c1] * fr * fc
            cr_val = cr_matrix[r0][c0] * (1 - fr) * (1 - fc) + cr_matrix[r0][c1] * (1 - fr) * fc + \
                     cr_matrix[r1][c0] * fr * (1 - fc) + cr_matrix[r1][c1] * fr * fc
            rgb = ycbcr2rgb((y_val, cb_val, cr_val))
            row_pixels.append(tuple(max(0, min(255, int(v))) for v in rgb))
        final_pixels.append(row_pixels)
    return final_pixels

def jpeg_encode_progressive_pixels(pixels, quality, subsampling='4:4:4'):
    """
    将像素编码为渐进式 JPEG。
    修复1：分量级霍夫曼表映射，Y用表0，Cb/Cr用表1。
    修复2：AC扫描段每个分量必须独立，不允许交织（JPEG标准规定）。
    修复3：合并亮度AC为1-63，避免非主流频段拆分导致兼容性问题。
    修复4：区分交织与非交织扫描段，非交织扫描段必须按光栅顺序排列块数据，修复子采样导致的错位和方块问题。
    """
    is_gray = len(pixels[0][0]) == 1
    subsampling_map = {'4:4:4': (1, 1), '4:2:2': (2, 1), '4:2:0': (2, 2), '4:1:1': (4, 1), '4:4:0': (1, 2), '4:1:0': (4, 2)}
    if isinstance(subsampling, str):
        yh, yv = subsampling_map.get(subsampling, (1, 1))
    else:
        yh, yv = subsampling

    if is_gray:
        yh, yv = (1, 1)
        y_matrix = [[p[0] if isinstance(p, (bytes, tuple, list)) else p for p in row] for row in pixels]
    else:
        split = tuple(map(list, zip(*[map(list, zip(*map(rgb2ycbcr, row))) for row in pixels])))
        y_matrix, cb_matrix, cr_matrix = split[0], split[1], split[2]

    height = len(y_matrix)
    width = len(y_matrix[0])
    target_h = (height + yv * 8 - 1) // (yv * 8) * yv * 8
    target_w = (width + yh * 8 - 1) // (yh * 8) * yh * 8

    def pad_matrix(matrix, t_h, t_w):
        h = len(matrix)
        w = len(matrix[0])
        padded = [row[:] for row in matrix]
        if w < t_w:
            for r in padded:
                r.extend([r[-1]] * (t_w - w))
        if h < t_h:
            last_row = padded[-1][:]
            padded.extend([last_row[:] for _ in range(t_h - h)])
        return padded

    y_matrix = pad_matrix(y_matrix, target_h, target_w)

    def downsample_and_pad(matrix, h_factor, v_factor, t_h, t_w):
        orig_h = len(matrix)
        orig_w = len(matrix[0])
        ds_h = (orig_h + v_factor - 1) // v_factor
        ds_w = (orig_w + h_factor - 1) // h_factor
        ds = [[0] * ds_w for _ in range(ds_h)]
        for r in range(ds_h):
            for c in range(ds_w):
                val = 0.0
                weight = 0.0
                for vr in range(v_factor):
                    for hc in range(h_factor):
                        orig_r = r * v_factor + vr
                        orig_c = c * h_factor + hc
                        if orig_r < orig_h and orig_c < orig_w:
                            val += matrix[orig_r][orig_c]
                            weight += 1.0
                ds[r][c] = round(val / weight) if weight > 0 else 0
        bh = (ds_h + 7) // 8 * 8
        bw = (ds_w + 7) // 8 * 8
        return pad_matrix(ds, bh, bw), bh, bw

    cb_padded = cr_padded = None
    if not is_gray:
        cb_padded, cb_bh, cb_bw = downsample_and_pad(cb_matrix, yh, yv, target_h, target_w)
        cr_padded, _, _ = downsample_and_pad(cr_matrix, yh, yv, target_h, target_w)

    lqtable = jpeg_adjust_qtable(_STD_LUMA_QTABLE, quality)
    cqtable = jpeg_adjust_qtable(_STD_CHROMA_QTABLE, quality)

    def process_blocks(matrix, qtable):
        rows = len(matrix)
        cols = len(matrix[0])
        b_rows = rows // 8
        b_cols = cols // 8
        all_zigzag = []
        for br in range(b_rows):
            row_zz = []
            for bc in range(b_cols):
                block = [[matrix[br * 8 + i][bc * 8 + j] for j in range(8)] for i in range(8)]
                for i in range(8):
                    for j in range(8):
                        block[i][j] -= 128
                block = jpeg_dct8x8(block)
                for i in range(8):
                    for j in range(8):
                        block[i][j] = round(block[i][j] / qtable[i][j])
                zz = jpeg_zigzag(block)
                row_zz.append(zz)
            all_zigzag.append(row_zz)
        return all_zigzag

    y_zz = process_blocks(y_matrix, lqtable)
    cb_zz = cr_zz = None
    if not is_gray:
        cb_zz = process_blocks(cb_padded, cqtable)
        cr_zz = process_blocks(cr_padded, cqtable)

    mcu_rows = target_h // (yv * 8)
    mcu_cols = target_w // (yh * 8)

    # ===== 扫描段定义（关键修复区域）=====
    scan_specs = []
    if is_gray:
        scan_specs.append({'comp_ids': [1], 'Ss': 0, 'Se': 0, 'Ah': 0, 'Al': 0, 'dc_tables': {1: 0}, 'ac_tables': {1: 0}})
        scan_specs.append({'comp_ids': [1], 'Ss': 1, 'Se': 63, 'Ah': 0, 'Al': 0, 'dc_tables': {1: 0}, 'ac_tables': {1: 0}})
    else:
        # 1. DC 扫描段：所有分量交织（JPEG标准允许DC交织）
        scan_specs.append({
            'comp_ids': [1, 2, 3],
            'Ss': 0, 'Se': 0, 'Ah': 0, 'Al': 0,
            'dc_tables': {1: 0, 2: 1, 3: 1},
            'ac_tables': {1: 0, 2: 0, 3: 0}
        })
        # 2. 亮度 Y AC（非交织，单分量）
        scan_specs.append({
            'comp_ids': [1],
            'Ss': 1, 'Se': 63, 'Ah': 0, 'Al': 0,
            'dc_tables': {1: 0}, 'ac_tables': {1: 0}
        })
        # 3. 色度 Cb AC（非交织，单分量）
        scan_specs.append({
            'comp_ids': [2],
            'Ss': 1, 'Se': 63, 'Ah': 0, 'Al': 0,
            'dc_tables': {2: 1}, 'ac_tables': {2: 1}
        })
        # 4. 色度 Cr AC（非交织，单分量）
        scan_specs.append({
            'comp_ids': [3],
            'Ss': 1, 'Se': 63, 'Ah': 0, 'Al': 0,
            'dc_tables': {3: 1}, 'ac_tables': {3: 1}
        })

    def get_block_zz(comp_id, r_mcu, c_mcu, vr, hc):
        if comp_id == 1:
            br = r_mcu * yv + vr
            bc = c_mcu * yh + hc
            return y_zz[br][bc][:]
        elif comp_id == 2:
            return cb_zz[r_mcu][c_mcu][:]
        else:
            return cr_zz[r_mcu][c_mcu][:]

    dc_huff_funcs = [jpeg_luma_dc_huff, jpeg_chroma_dc_huff]
    ac_huff_tables = __AC

    scan_data_list = []
    for spec in scan_specs:
        scan_comp_ids = spec['comp_ids']
        Ss = spec['Ss']
        Se = spec['Se']
        Ah = spec['Ah']
        Al = spec['Al']
        dc_tables_map = spec.get('dc_tables', {cid: spec.get('dc_table', 0) for cid in scan_comp_ids})
        ac_tables_map = spec.get('ac_tables', {cid: spec.get('ac_table', 0) for cid in scan_comp_ids})

        bin_str = ''
        prev_dc_local = {cid: 0 for cid in scan_comp_ids}
        
        is_interleaved = len(scan_comp_ids) > 1

        if is_interleaved:
            # === 交织扫描段（如 DC 扫描段）：按 MCU 顺序排列 ===
            for r_mcu in range(mcu_rows):
                for c_mcu in range(mcu_cols):
                    for comp_id in scan_comp_ids:
                        dc_table_id = dc_tables_map[comp_id]
                        ac_table_id = ac_tables_map[comp_id]
                        dc_huff = dc_huff_funcs[dc_table_id]
                        ac_huff = ac_huff_tables[ac_table_id]
                        v_samp = yv if comp_id == 1 else 1
                        h_samp = yh if comp_id == 1 else 1
                        for vr in range(v_samp):
                            for hc in range(h_samp):
                                zz = get_block_zz(comp_id, r_mcu, c_mcu, vr, hc)
                                if Ss == 0 and Se == 0:
                                    dc_val = zz[0]
                                    dpcm = dc_val - prev_dc_local[comp_id]
                                    prev_dc_local[comp_id] = dc_val
                                    length, value = jpeg_category(dpcm)
                                    bin_str += dc_huff(length) + value
                                else:
                                    ac_coeffs = zz[Ss:Se+1]
                                    rle = jpeg_rle_encoding(ac_coeffs)
                                    if not rle or (len(rle) == 1 and rle[0] == (0, 0)):
                                        bin_str += ac_huff[0, 0]
                                    else:
                                        encoded_rle = [(x, *jpeg_category(y)) for x, y in rle]
                                        for run, cat_len, cat_val in encoded_rle:
                                            if run == 0 and cat_len == 0:
                                                bin_str += ac_huff[0, 0]
                                            elif run == 15 and cat_len == 0:
                                                bin_str += ac_huff[15, 0]
                                            else:
                                                bin_str += ac_huff[run, cat_len] + cat_val
        else:
            # === 非交织扫描段（如 Y AC、Cb AC、Cr AC）：必须按光栅顺序排列 ===
            comp_id = scan_comp_ids[0]
            dc_table_id = dc_tables_map[comp_id]
            ac_table_id = ac_tables_map[comp_id]
            dc_huff = dc_huff_funcs[dc_table_id]
            ac_huff = ac_huff_tables[ac_table_id]
            
            # 获取该分量总块行数和总块列数
            if comp_id == 1:
                zz_matrix = y_zz
            elif comp_id == 2:
                zz_matrix = cb_zz
            else:
                zz_matrix = cr_zz
                
            total_b_rows = len(zz_matrix)
            total_b_cols = len(zz_matrix[0]) if total_b_rows > 0 else 0
            
            for br in range(total_b_rows):
                for bc in range(total_b_cols):
                    zz = zz_matrix[br][bc][:]
                    if Ss == 0 and Se == 0:
                        dc_val = zz[0]
                        dpcm = dc_val - prev_dc_local[comp_id]
                        prev_dc_local[comp_id] = dc_val
                        length, value = jpeg_category(dpcm)
                        bin_str += dc_huff(length) + value
                    else:
                        ac_coeffs = zz[Ss:Se+1]
                        rle = jpeg_rle_encoding(ac_coeffs)
                        if not rle or (len(rle) == 1 and rle[0] == (0, 0)):
                            bin_str += ac_huff[0, 0]
                        else:
                            encoded_rle = [(x, *jpeg_category(y)) for x, y in rle]
                            for run, cat_len, cat_val in encoded_rle:
                                if run == 0 and cat_len == 0:
                                    bin_str += ac_huff[0, 0]
                                elif run == 15 and cat_len == 0:
                                    bin_str += ac_huff[15, 0]
                                else:
                                    bin_str += ac_huff[run, cat_len] + cat_val

        if len(bin_str) % 8 != 0:
            bin_str += '1' * (8 - len(bin_str) % 8)
        data = int(bin_str, 2).to_bytes(len(bin_str) // 8, 'big').replace(b'\xff', b'\xff\x00')
        
        scan_data_list.append({
            'data': data,
            'comp_ids': scan_comp_ids,
            'Ss': Ss,
            'Se': Se,
            'Ah': Ah,
            'Al': Al,
            'dc_tables': dc_tables_map,
            'ac_tables': ac_tables_map
        })

    components = 1 if is_gray else 3
    return (scan_data_list, lqtable, cqtable if not is_gray else lqtable, yh, yv, components)
class JPEG(BaseImage):
    def read(self, filename: str) -> None:
        def build_huffman_tree(counts, values, table_class):
            root = [None, None]
            code = 0
            val_idx = 0
            for length in range(1, 17):
                for _ in range(counts[length - 1]):
                    node = root
                    for bit_idx in map(int, format(code, '0{}b'.format(length))):
                        if node[bit_idx] is None:
                            node[bit_idx] = [None, None]
                        node = node[bit_idx]
                    v = values[val_idx]
                    if table_class == 0:
                        node[0] = v
                        node[1] = v
                    else:
                        node[0] = v >> 4
                        node[1] = v & 15
                    val_idx += 1
                    code += 1
                code <<= 1
            return root

        q_tables = {}
        huff_tables = {'dc': {}, 'ac': {}}
        comp_ids = []
        comp_info = {}
        comp_huff = {}
        yh = yv = 1
        is_progressive = False
        progressive_scans = []

        # 使用内部函数封装读取逻辑
        def _parse_jpeg_data(rb):
            nonlocal yh, yv, is_progressive, progressive_scans
            nonlocal q_tables, huff_tables, comp_ids, comp_info, comp_huff
            
            if rb.read(2) != b'\xff\xd8':
                raise ValueError('Not a valid JPEG file: missing SOI marker (FFD8)')
                
            next_marker = None
            while True:
                # 优先使用上次扫描段读取中截获的标记
                if next_marker is not None:
                    marker_code = next_marker
                    next_marker = None
                else:
                    while rb.read(1) != b'\xff':
                        continue
                    while True:
                        marker_code = rb.read(1)
                        if not marker_code:
                            raise ValueError('Unexpected end of file after marker start byte (FF)')
                        marker_code = marker_code[0]
                        if marker_code == 255:
                            continue
                        if marker_code == 0 or 208 <= marker_code <= 215:
                            break
                        break
                        
                if marker_code == 0 or 208 <= marker_code <= 215:
                    continue
                    
                if marker_code in (0xC0, 0xC2):
                    segment_length = int.from_bytes(rb.read(2), 'big')
                    segment_data = rb.read(segment_length - 2)
                    is_progressive = (marker_code == 0xC2)
                    self._bit_depth = segment_data[0]
                    self._height = int.from_bytes(segment_data[1:3], 'big')
                    self._width = int.from_bytes(segment_data[3:5], 'big')
                    components = segment_data[5]
                    if components == 3:
                        self._color_mode = 'RGB'
                    elif components == 1:
                        self._color_mode = 'GRAY'
                    else:
                        self._color_mode = 'RGB'
                    comp_ids = []
                    comp_info = {}
                    for i in range(components):
                        c_id = segment_data[6 + i * 3]
                        sampling = segment_data[7 + i * 3]
                        h_samp = sampling >> 4
                        v_samp = sampling & 15
                        q_id = segment_data[8 + i * 3]
                        comp_ids.append(c_id)
                        comp_info[c_id] = {'h': h_samp, 'v': v_samp, 'q_id': q_id}
                        if i == 0:
                            yh = h_samp
                            yv = v_samp
                    continue
                    
                if marker_code == 0xDB:
                    segment_length = int.from_bytes(rb.read(2), 'big')
                    segment_data = rb.read(segment_length - 2)
                    i = 0
                    while i < len(segment_data):
                        info_byte = segment_data[i]
                        precision = info_byte >> 4 & 15
                        table_id = info_byte & 15
                        i += 1
                        if precision != 0:
                            raise NotImplementedError('16-bit quantization tables are not supported')
                        table_data = jpeg_zigzag(list(segment_data[i:i + 64]), True)
                        i += 64
                        q_tables[table_id] = table_data
                    continue
                    
                if marker_code == 0xC4:
                    segment_length = int.from_bytes(rb.read(2), 'big')
                    segment_data = rb.read(segment_length - 2)
                    i = 0
                    while i < len(segment_data):
                        info_byte = segment_data[i]
                        table_class = info_byte >> 4 & 15
                        table_id = info_byte & 15
                        i += 1
                        counts = list(segment_data[i:i + 16])
                        i += 16
                        total_values = sum(counts)
                        values = list(segment_data[i:i + total_values])
                        i += total_values
                        tree = build_huffman_tree(counts, values, table_class)
                        if table_class == 0:
                            huff_tables['dc'][table_id] = tree
                        else:
                            huff_tables['ac'][table_id] = tree
                    continue
                    
                if marker_code == 0xDD:
                    segment_length = int.from_bytes(rb.read(2), 'big')
                    rb.read(segment_length - 2)
                    continue
                    
                if marker_code == 0xDA:
                    segment_length = int.from_bytes(rb.read(2), 'big')
                    sos_data = rb.read(segment_length - 2)
                    num_comp = sos_data[0]
                    scan_comp_ids = []
                    scan_comp_huff = {}
                    for i in range(num_comp):
                        c_id = sos_data[1 + i * 2]
                        tables = sos_data[2 + i * 2]
                        scan_comp_ids.append(c_id)
                        scan_comp_huff[c_id] = {'dc': tables >> 4, 'ac': tables & 15}
                        comp_huff[c_id] = {'dc': tables >> 4, 'ac': tables & 15}
                    Ss = sos_data[1 + num_comp * 2]
                    Se = sos_data[2 + num_comp * 2]
                    Ah_Al = sos_data[3 + num_comp * 2]
                    Ah = Ah_Al >> 4
                    Al = Ah_Al & 15
                    
                    scan_bytes = bytearray()
                    rst_offsets = []
                    byte_count = 0
                    prev_ff = False
                    while True:
                        byte = rb.read(1)
                        if not byte:
                            break
                        b = byte[0]
                        if prev_ff:
                            if b == 0:
                                scan_bytes.append(255)
                                byte_count += 1
                                prev_ff = False
                            elif 208 <= b <= 215:
                                rst_offsets.append(byte_count)
                                prev_ff = False
                            elif b == 255:
                                pass
                            else:
                                # 遇到下一个标记，保存它并结束当前扫描段读取
                                next_marker = b
                                break
                        elif b == 255:
                            prev_ff = True
                        else:
                            scan_bytes.append(b)
                            byte_count += 1
                            
                    if not is_progressive:
                        pixels = jpeg_decode_pixels(
                            bytes(scan_bytes), q_tables, huff_tables, self._width, self._height,
                            yh, yv, comp_ids, comp_info, comp_huff, rst_offsets)
                        self._pixels = [[bytes(pixel) for pixel in row] for row in pixels]
                    else:
                        progressive_scans.append({
                            'data': bytes(scan_bytes),
                            'comp_ids': scan_comp_ids,
                            'Ss': Ss,
                            'Se': Se,
                            'Ah': Ah,
                            'Al': Al,
                            'rst_offsets': rst_offsets,
                        })
                    continue
                    
                if marker_code == 0xD9:
                    # 读到 EOI，内部函数直接 return 退出，回到外层函数
                    return
                    
                segment_length = int.from_bytes(rb.read(2), 'big')
                rb.read(segment_length - 2)

        with open(filename, 'rb') as rb:
            _parse_jpeg_data(rb)
            
        # 内部函数退出后，检查是否为渐进式图像，如果是，统一解码储存像素数据
        if is_progressive and progressive_scans:
            pixels = jpeg_decode_progressive_pixels(
                progressive_scans, q_tables, huff_tables, self._width, self._height,
                yh, yv, comp_ids, comp_info, comp_huff)
            self._pixels = [[bytes(pixel) for pixel in row] for row in pixels]

    def write(self, filename: str = None, quality=50, subsampling='4:4:4', progressive=False) -> bytes:
        if self._pixels is None:
            raise ValueError('Please create or read an image first')
        pixel_matrix = []
        for row in self._pixels:
            pixel_row = []
            for pixel in row:
                pixel_row.append(pixel)
            pixel_matrix.append(pixel_row)

        if not progressive:
            encoded, lqtable, cqtable, yh, yv, components = jpeg_encode_pixels(pixel_matrix, quality, subsampling)
            is_gray = components == 1
            dqt = b'\xff\xdb\x00C\x00' + bytes(jpeg_zigzag(lqtable))
            if not is_gray:
                dqt += b'\xff\xdb\x00C\x01' + bytes(jpeg_zigzag(cqtable))

            if is_gray:
                sof0 = b''.join([
                    b'\xff\xc0\x00\x0b\x08',
                    self._height.to_bytes(2, 'big'),
                    self._width.to_bytes(2, 'big'),
                    b'\x01',
                    b'\x01' + bytes([yh << 4 | yv]) + b'\x00'])
            else:
                sof0 = b''.join([
                    b'\xff\xc0\x00\x11\x08',
                    self._height.to_bytes(2, 'big'),
                    self._width.to_bytes(2, 'big'),
                    b'\x03',
                    b'\x01' + bytes([yh << 4 | yv]) + b'\x00',
                    b'\x02\x11\x01',
                    b'\x03\x11\x01'])

            dht = _DHT_DC0 + _DHT_AC0
            if not is_gray:
                dht += _DHT_DC1 + _DHT_AC1

            if is_gray:
                sos = b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00'
            else:
                sos = b'\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00'

            image = b''.join([
                b'\xff\xd8',
                b'\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00',
                dqt, sof0, dht, sos, encoded,
                b'\xff\xd9'])
        else:
            scan_data_list, lqtable, cqtable, yh, yv, components = \
                jpeg_encode_progressive_pixels(pixel_matrix, quality, subsampling)
            is_gray = components == 1
            dqt = b'\xff\xdb\x00C\x00' + bytes(jpeg_zigzag(lqtable))
            if not is_gray:
                dqt += b'\xff\xdb\x00C\x01' + bytes(jpeg_zigzag(cqtable))

            if is_gray:
                sof = b''.join([
                    b'\xff\xc2\x00\x0b\x08',
                    self._height.to_bytes(2, 'big'),
                    self._width.to_bytes(2, 'big'),
                    b'\x01',
                    b'\x01' + bytes([yh << 4 | yv]) + b'\x00'])
            else:
                sof = b''.join([
                    b'\xff\xc2\x00\x11\x08',
                    self._height.to_bytes(2, 'big'),
                    self._width.to_bytes(2, 'big'),
                    b'\x03',
                    b'\x01' + bytes([yh << 4 | yv]) + b'\x00',
                    b'\x02\x11\x01',
                    b'\x03\x11\x01'])

            dht = _DHT_DC0 + _DHT_AC0
            if not is_gray:
                dht += _DHT_DC1 + _DHT_AC1

            scan_parts = []
            for scan in scan_data_list:
                scan_comp_ids = scan['comp_ids']
                Ss = scan['Ss']
                Se = scan['Se']
                Ah = scan['Ah']
                Al = scan['Al']
                
                # 提取分量级表映射
                dc_tables_map = scan.get('dc_tables', {cid: scan.get('dc_table', 0) for cid in scan_comp_ids})
                ac_tables_map = scan.get('ac_tables', {cid: scan.get('ac_table', 0) for cid in scan_comp_ids})
                
                num_comp = len(scan_comp_ids)
                sos_header_len = 6 + 2 * num_comp
                sos_header = bytearray()
                sos_header.extend(sos_header_len.to_bytes(2, 'big'))
                sos_header.append(num_comp)
                for cid in scan_comp_ids:
                    sos_header.append(cid)
                    # === 核心修复：从映射字典中读取对应的表 ID ===
                    dc_id = dc_tables_map.get(cid, 0)
                    ac_id = ac_tables_map.get(cid, 0)
                    sos_header.append((dc_id << 4) | ac_id)
                    # ==============================================
                sos_header.append(Ss)
                sos_header.append(Se)
                sos_header.append((Ah << 4) | Al)
                scan_parts.append(b'\xff\xda' + bytes(sos_header) + scan['data'])

            image = b''.join([
                b'\xff\xd8',
                b'\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00',
                dqt, sof, dht] + scan_parts + [
                b'\xff\xd9'])

        if filename:
            with open(filename, 'wb') as wb:
                wb.write(image)
        return image
class OctreeNode:
    __slots__ = ['r_sum', 'g_sum', 'b_sum', 'count', 'children', 'is_leaf', 'palette_index', 'parent', 'level']

    def __init__(self, level=0, parent=None):
        self.r_sum = self.g_sum = self.b_sum = self.count = 0
        self.children = [None] * 8
        self.is_leaf = False
        self.palette_index = -1
        self.parent = parent
        self.level = level

    def find_index_greedy(self, r, g, b):
        if self.is_leaf:
            return self.palette_index
        idx = (r >> 7 - self.level & 1) << 2 | (g >> 7 - self.level & 1) << 1 | b >> 7 - self.level & 1
        if self.children[idx]:
            return self.children[idx].find_index_greedy(r, g, b)
        for c in self.children:
            if c:
                return c.find_index_greedy(r, g, b)
        return self.palette_index


def octree_quantize(pixels, num_colors=256):
    root = OctreeNode(level=0)
    level_nodes = [[] for _ in range(8)]
    pixel_counter = collections.Counter(pixels)
    unique_pixels = list(pixel_counter.keys())

    def add_color(node, r, g, b, level, count=1):
        if level == 8:
            node.is_leaf = True
            node.r_sum += r * count
            node.g_sum += g * count
            node.b_sum += b * count
            node.count += count
            return
        idx = (r >> 7 - level & 1) << 2 | (g >> 7 - level & 1) << 1 | b >> 7 - level & 1
        if not node.children[idx]:
            node.children[idx] = OctreeNode(level=level + 1, parent=node)
            if level < 7:
                level_nodes[level + 1].append(node.children[idx])
        add_color(node.children[idx], r, g, b, level + 1, count)

    for r, g, b in unique_pixels:
        add_color(root, r, g, b, 0, pixel_counter[r, g, b])
    total_leaf_count = 0

    def count_leaves(node):
        nonlocal total_leaf_count
        if node.is_leaf:
            total_leaf_count += 1
            return
        for child in node.children:
            if child is not None:
                count_leaves(child)

    count_leaves(root)
    reducible = []

    def is_reducible(node):
        if node.is_leaf:
            return False
        for child in node.children:
            if child is not None and (not child.is_leaf):
                return False
        return True

    for level in range(7, 0, -1):
        for node in level_nodes[level]:
            if is_reducible(node):
                heapq.heappush(reducible, (node.count, id(node), node))
    while total_leaf_count > num_colors and reducible:
        _, _, node = heapq.heappop(reducible)
        if not is_reducible(node):
            continue
        child_count = sum((1 for c in node.children if c is not None))
        if total_leaf_count - (child_count - 1) < num_colors:
            break
        for i in range(8):
            child = node.children[i]
            if child is not None:
                node.r_sum += child.r_sum
                node.g_sum += child.g_sum
                node.b_sum += child.b_sum
                node.count += child.count
                node.children[i] = None
        node.is_leaf = True
        total_leaf_count -= child_count - 1
        parent = node.parent
        if parent is not None and is_reducible(parent):
            heapq.heappush(reducible, (parent.count, id(parent), parent))
    leaves = []

    def collect_current_leaves(node):
        if node.is_leaf:
            leaves.append(node)
            return
        for child in node.children:
            if child is not None:
                collect_current_leaves(child)

    collect_current_leaves(root)
    if total_leaf_count > num_colors:
        merge_costs = []
        for i, n1 in enumerate(leaves):
            r1, g1, b1 = (n1.r_sum / n1.count, n1.g_sum / n1.count, n1.b_sum / n1.count)
            for j, n2 in enumerate(leaves[i + 1:], i + 1):
                r2, g2, b2 = (n2.r_sum / n2.count, n2.g_sum / n2.count, n2.b_sum / n2.count)
                dist = ((r1 - r2) * 0.3) ** 2 + ((g1 - g2) * 0.59) ** 2 + ((b1 - b2) * 0.11) ** 2
                merge_costs.append((dist, i, j))
        merge_costs.sort(key=lambda x: x[0])
        removed = set()
        for cost, i, j in merge_costs:
            if total_leaf_count <= num_colors:
                break
            n1 = leaves[i]
            n2 = leaves[j]
            if id(n1) in removed or id(n2) in removed:
                continue
            n1.r_sum += n2.r_sum
            n1.g_sum += n2.g_sum
            n1.b_sum += n2.b_sum
            n1.count += n2.count
            removed.add(id(n2))
            total_leaf_count -= 1
        leaves = [n for n in leaves if id(n) not in removed]
    palette = []
    for leaf in leaves:
        if leaf.count > 0:
            palette.append(
                (round(leaf.r_sum / leaf.count), round(leaf.g_sum / leaf.count), round(leaf.b_sum / leaf.count)))
    while len(palette) < num_colors:
        palette.append((0, 0, 0))
    clean_root = OctreeNode(level=0)
    for idx, (r, g, b) in enumerate(palette):
        node = clean_root
        for level in range(8):
            bit_idx = (r >> 7 - level & 1) << 2 | (g >> 7 - level & 1) << 1 | b >> 7 - level & 1
            if node.children[bit_idx] is None:
                node.children[bit_idx] = OctreeNode(level=level + 1, parent=node)
            node = node.children[bit_idx]
        node.is_leaf = True
        node.palette_index = idx
    return (palette, clean_root)


class _BitReader:
    __slots__ = ('data', 'byte_pos', 'buf', 'bits_in')

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte_pos = 0
        self.buf = 0
        self.bits_in = 0

    def read(self, nbits: int) -> int:
        while self.bits_in < nbits:
            if self.byte_pos >= len(self.data):
                raise ValueError('BitReader: unexpected end of data')
            self.buf |= self.data[self.byte_pos] << self.bits_in
            self.byte_pos += 1
            self.bits_in += 8
        code = self.buf & (1 << nbits) - 1
        self.buf >>= nbits
        self.bits_in -= nbits
        return code


class _BitWriter:
    __slots__ = ('buf', 'bits_in', 'out')

    def __init__(self) -> None:
        self.buf = 0
        self.bits_in = 0
        self.out = bytearray()

    def write(self, code: int, nbits: int) -> None:
        self.buf |= code << self.bits_in
        self.bits_in += nbits
        while self.bits_in >= 8:
            self.out.append(self.buf & 255)
            self.buf >>= 8
            self.bits_in -= 8

    def flush(self) -> bytes:
        if self.bits_in > 0:
            self.out.append(self.buf & 255)
            self.buf = 0
            self.bits_in = 0
        result = bytes(self.out)
        self.out.clear()
        return result


def _lzw_decode(min_code_size: int, data: bytes) -> list:
    if min_code_size < 2:
        min_code_size = 2
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1
    table = {i: [i] for i in range(clear_code)}
    next_code = eoi_code + 1
    code_size = min_code_size + 1
    max_code = (1 << code_size) - 1
    reader = _BitReader(data)
    output = []
    code = reader.read(code_size)
    if code != clear_code:
        raise ValueError('LZW: first code should be clear_code')
    code = reader.read(code_size)
    if code == eoi_code:
        return output
    if code not in table:
        raise ValueError('LZW: invalid first code')
    prev = table[code][:]
    output.extend(prev)
    while True:
        try:
            code = reader.read(code_size)
        except ValueError:
            break
        if code == eoi_code:
            break
        if code == clear_code:
            table = {i: [i] for i in range(clear_code)}
            next_code = eoi_code + 1
            code_size = min_code_size + 1
            max_code = (1 << code_size) - 1
            code = reader.read(code_size)
            if code == eoi_code:
                break
            if code not in table:
                raise ValueError('LZW: invalid code after clear')
            prev = table[code][:]
            output.extend(prev)
            continue
        if code in table:
            entry = table[code]
        elif code == next_code:
            entry = prev + [prev[0]]
        else:
            raise ValueError('LZW: bad code')
        output.extend(entry)
        if next_code < 4096:
            table[next_code] = prev + [entry[0]]
            next_code += 1
            if next_code > max_code and code_size < 12:
                code_size += 1
                max_code = (1 << code_size) - 1
        prev = entry[:]
    return output


def _lzw_encode(min_code_size: int, pixels: list) -> bytes:
    if min_code_size < 2:
        min_code_size = 2
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1
    writer = _BitWriter()
    code_size = min_code_size + 1
    max_code = (1 << code_size) - 1
    enc_next = eoi_code + 1
    dec_next = eoi_code + 1
    dec_first = True
    table = {(i,): i for i in range(clear_code)}

    def _do_reset() -> None:
        nonlocal code_size, max_code, enc_next, dec_next, dec_first, table
        code_size = min_code_size + 1
        max_code = (1 << code_size) - 1
        enc_next = eoi_code + 1
        dec_next = eoi_code + 1
        dec_first = True
        table = {(i,): i for i in range(clear_code)}
        writer.write(clear_code, code_size)

    if not pixels:
        writer.write(clear_code, code_size)
        writer.write(eoi_code, code_size)
        return writer.flush()
    current = (pixels[0],)

    def _dec_step() -> None:
        nonlocal dec_first, dec_next, code_size, max_code
        if dec_first:
            dec_first = False
            return
        if dec_next < 4096:
            dec_next += 1
            if dec_next > max_code and code_size < 12:
                code_size += 1
                max_code = (1 << code_size) - 1

    for i in range(1, len(pixels)):
        p = pixels[i]
        candidate = current + (p,)
        if candidate in table:
            current = candidate
        else:
            writer.write(table[current], code_size)
            _dec_step()
            if enc_next < 4096:
                table[candidate] = enc_next
                enc_next += 1
            else:
                writer.write(clear_code, code_size)
                _do_reset()
            current = (p,)
    writer.write(table[current], code_size)
    _dec_step()
    writer.write(eoi_code, code_size)
    return writer.flush()


def _read_sub_blocks(data: bytes, offset: int) -> tuple:
    result = bytearray()
    while True:
        if offset >= len(data):
            raise ValueError('Unexpected end of sub-blocks')
        block_size = data[offset]
        offset += 1
        if block_size == 0:
            break
        result.extend(data[offset:offset + block_size])
        offset += block_size
    return (bytes(result), offset)


def _write_sub_blocks(raw: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(raw):
        chunk = raw[i:i + 255]
        out.append(len(chunk))
        out.extend(chunk)
        i += 255
    out.append(0)
    return bytes(out)


_INTERLACE_PASSES = [(0, 8), (4, 8), (2, 4), (1, 2)]


def _deinterlace(pixels: list, w: int, h: int) -> list:
    rows = [None] * h
    idx = 0
    for start, step in _INTERLACE_PASSES:
        for y in range(start, h, step):
            rows[y] = pixels[idx:idx + w]
            idx += w
    flat = []
    for row in rows:
        flat.extend(row)
    return flat


def _interlace(pixels: list, w: int, h: int) -> list:
    rows = [pixels[y * w:(y + 1) * w] for y in range(h)]
    flat = []
    for start, step in _INTERLACE_PASSES:
        for y in range(start, h, step):
            flat.extend(rows[y])
    return flat


def _norm_palette(pal: list) -> list:
    n = min(len(pal), 256)
    size = 1
    while size < n:
        size <<= 1
    size = max(size, 2)
    return list(pal[:n]) + [(0, 0, 0)] * (size - n)


class GIF(BaseImage):

    def __init__(self) -> None:
        super().__init__()
        self._frames = []
        self._loop_count = 0
        self._comments = []
        self._bg_index = 0
        self._pixel_aspect = 0
        self._color_resolution = 7
        self._canvas = None
        self._saved_canvas = None

    def new(self, width: int, height: int, color: tuple = (), color_mode: str = 'INDEX', bit_depth: int = 8,
            palette: list = None, left: int = 0, top: int = 0, delay_cs: int = 0, disposal: int = 0,
            has_transparent: bool = False, transparent_index: int = 0, interlace: bool = False,
            local_palette: list = None) -> None:
        color_mode = color_mode.strip().upper()
        if color_mode != 'INDEX':
            raise ValueError('GIF only supports INDEX color mode')
        if bit_depth not in (1, 2, 4, 8):
            raise ValueError('GIF supports bit_depth of 1, 2, 4, or 8')
        super().new(width, height, color, color_mode, bit_depth, palette)
        frame_pixels = list(map(list, self._pixels))
        frame = {'pixels': frame_pixels, 'width': width, 'height': height, 'left': left, 'top': top,
                 'interlace': interlace,
                 'gce': {'disposal': disposal & 7, 'delay_cs': delay_cs, 'has_transparent': 1 if has_transparent else 0,
                         'transparent_index': transparent_index, 'user_input': 0}, 'lct': local_palette}
        self._frames = [frame]
        self._canvas = None

    def add_frame(self, img: BaseImage, left: int = 0, top: int = 0, delay_cs: int = 0, disposal: int = 0,
                  has_transparent: bool = False, transparent_index: int = 0, interlace: bool = False) -> None:
        if not isinstance(img, BaseImage):
            raise TypeError('Argument must be a BaseImage instance')
        if img.color_mode != 'INDEX':
            raise ValueError('GIF only supports INDEX color mode images. Please convert the image first.')
        frame_pixels = list(map(list, img._pixels))
        local_palette = img.palette
        frame = {'pixels': frame_pixels, 'width': img.width, 'height': img.height, 'left': left, 'top': top,
                 'interlace': interlace,
                 'gce': {'disposal': disposal & 7, 'delay_cs': delay_cs, 'has_transparent': 1 if has_transparent else 0,
                         'transparent_index': transparent_index, 'user_input': 0}, 'lct': local_palette}
        self._frames.append(frame)
        self._canvas = None

    def __getitem__(self, item: Any) -> tuple:
        if isinstance(item, int):
            if item < 0 or item >= len(self._frames):
                raise IndexError('Frame index out of range')
            frame_data = self._frames[item]
            view = BaseImage()
            view._width = frame_data['width']
            view._height = frame_data['height']
            view._color_mode = 'INDEX'
            view._bit_depth = self._bit_depth
            view._palette = frame_data['lct'] if frame_data['lct'] else self._palette
            view._pixels = frame_data['pixels']
            return view
        elif isinstance(item, tuple) and len(item) == 2:
            if not self._frames:
                raise IndexError('No frames')
            x, y = item
            p = self._frames[0]['pixels'][y][x]
            return (p[0],)
        raise TypeError('Invalid index type')

    def __setitem__(self, key: tuple, value: tuple) -> None:
        if isinstance(key, tuple) and len(key) == 2:
            if not self._frames:
                raise IndexError('No frames')
            x, y = key
            if not isinstance(value, tuple) or len(value) != 1:
                raise ValueError('Value must be a 1-tuple (index,)')
            idx = value[0]
            self._frames[0]['pixels'][y][x] = bytes([idx])
        else:
            raise TypeError('Invalid index type')

    def read(self, filename: str) -> None:
        with open(filename, 'rb') as f:
            d = f.read()
        sig = d[0:3].decode('ascii')
        ver = d[3:6].decode('ascii')
        if sig != 'GIF':
            raise ValueError('Not a GIF file')
        if ver not in ('87a', '89a'):
            raise ValueError('Unsupported GIF version: ' + ver)
        offset = 6
        self._width = int.from_bytes(d[offset:offset + 2], 'little')
        offset += 2
        self._height = int.from_bytes(d[offset:offset + 2], 'little')
        offset += 2
        packed = d[offset]
        offset += 1
        self._bg_index = d[offset]
        offset += 1
        self._pixel_aspect = d[offset]
        offset += 1
        gct_flag = packed >> 7 & 1
        self._color_resolution = packed >> 4 & 7
        gct_sf = packed & 7
        self._color_mode = 'INDEX'
        self._bit_depth = gct_sf + 1 if gct_flag else self._color_resolution + 1
        gct = []
        if gct_flag:
            n = 2 ** (gct_sf + 1)
            for _ in range(n):
                r, g, b = (d[offset], d[offset + 1], d[offset + 2])
                offset += 3
                gct.append((r, g, b))
        self._palette = gct
        self._frames = []
        self._comments = []
        self._loop_count = 0
        pending_gce = None
        while offset < len(d):
            b = d[offset]
            offset += 1
            if b == 59:
                break
            elif b == 44:
                left = int.from_bytes(d[offset:offset + 2], 'little')
                offset += 2
                top = int.from_bytes(d[offset:offset + 2], 'little')
                offset += 2
                fw = int.from_bytes(d[offset:offset + 2], 'little')
                offset += 2
                fh = int.from_bytes(d[offset:offset + 2], 'little')
                offset += 2
                packed = d[offset]
                offset += 1
                lct_flag = packed >> 7 & 1
                interlace = bool(packed >> 6 & 1)
                lct_sf = packed & 7
                lct = []
                if lct_flag:
                    n = 2 ** (lct_sf + 1)
                    for _ in range(n):
                        r, g, b = (d[offset], d[offset + 1], d[offset + 2])
                        offset += 3
                        lct.append((r, g, b))
                mcs = d[offset]
                offset += 1
                comp, offset = _read_sub_blocks(d, offset)
                indices = _lzw_decode(mcs, comp)
                exp = fw * fh
                if len(indices) < exp:
                    indices.extend([0] * (exp - len(indices)))
                elif len(indices) > exp:
                    indices = indices[:exp]
                if interlace:
                    indices = _deinterlace(indices, fw, fh)
                pixels_matrix = [[bytes([indices[y * fw + x]]) for x in range(fw)] for y in range(fh)]
                self._frames.append({'pixels': pixels_matrix, 'width': fw, 'height': fh, 'left': left, 'top': top,
                                     'interlace': interlace,
                                     'gce': pending_gce if pending_gce else {'disposal': 0, 'delay_cs': 0,
                                                                             'has_transparent': 0,
                                                                             'transparent_index': 0, 'user_input': 0},
                                     'lct': lct})
                pending_gce = None
            elif b == 33:
                label = d[offset]
                offset += 1
                if label == 249:
                    gce = {}
                    block_size = d[offset]
                    offset += 1
                    if block_size != 4:
                        raise ValueError('Invalid GCE block size: expected 4, got ' + str(block_size))
                    pk = d[offset]
                    offset += 1
                    gce['disposal'] = pk >> 2 & 7
                    gce['user_input'] = pk >> 1 & 1
                    gce['has_transparent'] = pk & 1
                    gce['delay_cs'] = int.from_bytes(d[offset:offset + 2], 'little')
                    offset += 2
                    gce['transparent_index'] = d[offset]
                    offset += 1
                    offset += 1
                    pending_gce = gce
                elif label == 254:
                    content, offset = _read_sub_blocks(d, offset)
                    self._comments.append(content.decode('ascii', errors='replace'))
                elif label == 255:
                    block_size = d[offset]
                    offset += 1
                    if block_size != 11:
                        raise ValueError(
                            'Invalid Application Extension block size: expected 11, got ' + str(block_size))
                    app_id = d[offset:offset + 8].decode('ascii', errors='replace')
                    auth = d[offset + 8:offset + 11]
                    offset += 11
                    ad, offset = _read_sub_blocks(d, offset)
                    if app_id == 'NETSCAPE' and auth == b'2.0' and (len(ad) >= 3) and (ad[0] == 1):
                        self._loop_count = int.from_bytes(ad[1:3], 'little')
                    else:
                        bs = d[offset]
                        offset += 1
                        offset += bs
                        _, offset = _read_sub_blocks(d, offset)
                else:
                    bs = d[offset]
                    offset += 1
                    offset += bs
                    _, offset = _read_sub_blocks(d, offset)
        if self._frames:
            self._width = self._frames[0]['width']
            self._height = self._frames[0]['height']
            self._pixels = self._frames[0]['pixels']
        self._canvas = None

    def write(self, filename: str = None) -> bytes:
        if not self._frames:
            raise ValueError('No frames to write')
        out = bytearray()
        out.extend(b'GIF89a')
        out.extend(self._width.to_bytes(2, 'little'))
        out.extend(self._height.to_bytes(2, 'little'))
        gct_size_field = 0
        if self._palette:
            pal_len = min(len(self._palette), 256)
            while 1 << gct_size_field + 1 < pal_len:
                gct_size_field += 1
            packed = 128 | (self._color_resolution & 7) << 4 | gct_size_field & 7
        else:
            packed = (self._color_resolution & 7) << 4
        out.append(packed)
        out.append(self._bg_index)
        out.append(self._pixel_aspect)
        if self._palette:
            norm_gct = _norm_palette(self._palette)
            for r, g, b in norm_gct:
                out.extend([r, g, b])
        if self._loop_count is not None and len(self._frames) > 1:
            out.append(33)
            out.append(255)
            out.append(11)
            out.extend(b'NETSCAPE2.0')
            out.append(3)
            out.append(1)
            out.extend(self._loop_count.to_bytes(2, 'little'))
            out.append(0)
        for frame in self._frames:
            gce = frame['gce']
            if gce and (gce['delay_cs'] or gce['has_transparent'] or gce['disposal'] or gce.get('user_input', 0)):
                out.append(33)
                out.append(249)
                out.append(4)
                pk = (gce['disposal'] & 7) << 2
                if gce.get('user_input', 0):
                    pk |= 2
                if gce['has_transparent']:
                    pk |= 1
                out.append(pk)
                out.extend(gce['delay_cs'].to_bytes(2, 'little'))
                out.append(gce['transparent_index'])
                out.append(0)
            out.append(44)
            out.extend(frame['left'].to_bytes(2, 'little'))
            out.extend(frame['top'].to_bytes(2, 'little'))
            out.extend(frame['width'].to_bytes(2, 'little'))
            out.extend(frame['height'].to_bytes(2, 'little'))
            lct = frame['lct']
            pk = 0
            if lct:
                lct_sf = 0
                lct_len = min(len(lct), 256)
                while 1 << lct_sf + 1 < lct_len:
                    lct_sf += 1
                pk |= 128 | lct_sf & 7
            if frame['interlace']:
                pk |= 64
            out.append(pk)
            if lct:
                norm_lct = _norm_palette(lct)
                for r, g, b in norm_lct:
                    out.extend([r, g, b])
            fw, fh = (frame['width'], frame['height'])
            pixels_matrix = frame['pixels']
            flat_pix = [p[0] for row in pixels_matrix for p in row]
            if frame['interlace']:
                flat_pix = _interlace(flat_pix, fw, fh)
            pal = lct if lct else self._palette
            mcs = 2
            if pal:
                while 1 << mcs < len(pal):
                    mcs += 1
            mcs = max(mcs, 2)
            compressed = _lzw_encode(mcs, flat_pix)
            out.append(mcs)
            out.extend(_write_sub_blocks(compressed))
        out.append(59)
        image = bytes(out)
        if filename:
            with open(filename, 'wb') as wb:
                wb.write(image)
        return image

    def render_frame(self, frame_index: int) -> BaseImage:
        if frame_index < 0 or frame_index >= len(self._frames):
            raise IndexError('Frame index {} out of range'.format(frame_index))
        if frame_index == 0 or self._canvas is None:
            self._init_canvas()
        frame = self._frames[frame_index]
        if frame['gce']['disposal'] == 3:
            self._saved_canvas = list(map(list, self._canvas))
        self._draw_frame_to_canvas(frame)
        result_img = BaseImage()
        result_img.new(self._width, self._height, color_mode='RGBA', bit_depth=8)
        result_img._pixels = list(map(list, self._canvas))
        disposal = frame['gce']['disposal']
        if disposal == 2:
            self._restore_area_to_bg(frame['left'], frame['top'], frame['width'], frame['height'])
        elif disposal == 3:
            self._canvas = list(map(list, self._saved_canvas))
        return result_img

    def _init_canvas(self) -> None:
        bg = (0, 0, 0, 255)
        if self._palette and 0 <= self._bg_index < len(self._palette):
            bg = (*self._palette[self._bg_index][:3], 255)
        bg_bytes = bytes(bg)
        self._canvas = [[bg_bytes] * self._width for _ in range(self._height)]
        self._saved_canvas = None

    def _draw_frame_to_canvas(self, frame: dict) -> None:
        pixels = frame['pixels']
        fw, fh = (frame['width'], frame['height'])
        left, top = (frame['left'], frame['top'])
        pal = frame['lct'] if frame['lct'] else self._palette
        ht = frame['gce']['has_transparent']
        ti = frame['gce']['transparent_index'] if ht else -1
        cw, ch = (self._width, self._height)
        for fy in range(fh):
            cy = top + fy
            if cy < 0 or cy >= ch:
                continue
            row_pixels = pixels[fy]
            for fx in range(fw):
                cx = left + fx
                if cx < 0 or cx >= cw:
                    continue
                idx = row_pixels[fx][0]
                if ht and idx == ti:
                    continue
                if pal and 0 <= idx < len(pal):
                    r, g, b = pal[idx][:3]
                    self._canvas[cy][cx] = bytes([r, g, b, 255])

    def _restore_area_to_bg(self, left: int, top: int, width: int, height: int) -> None:
        bg = (0, 0, 0, 255)
        if self._palette and 0 <= self._bg_index < len(self._palette):
            bg = (*self._palette[self._bg_index][:3], 255)
        bg_bytes = bytes(bg)
        cw, ch = (self._width, self._height)
        start_y = max(0, top)
        end_y = min(ch, top + height)
        start_x = max(0, left)
        end_x = min(cw, left + width)
        w = end_x - start_x
        if w <= 0:
            return
        bg_row = [bg_bytes] * w
        for y in range(start_y, end_y):
            self._canvas[y][start_x:end_x] = bg_row
