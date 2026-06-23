import math
from .tools import linspace
from .types import Union, arr, real

thing = Union[list, str]


def colortext(text: str, rgb: arr) -> str:
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        raise ValueError('RGB must be a triplet')
    red, green, blue = rgb
    if not (0 <= red <= 255 and 0 <= green <= 255 and 0 <= blue <= 255):
        raise ValueError('The valid range for RGB values is from 0 to 255')
    number = 16 + round(blue / 51.2) * 36 + round(green / 51.2) * 6 + round(red / 51.2)
    return '\x1b[38;5;{}m{}\x1b[m'.format(number, text)


def safe_eval(func, x, y, threshold):
    try:
        val = func(x, y) - threshold
        if isinstance(val, complex):
            if abs(val.imag) < 1e-12:
                return float(val.real)
            return float('nan')
        return float(val)
    except (ValueError, ArithmeticError):
        return float('nan')


def estimate_lipschitz(func, domain, threshold, n=400, safety=1.5):
    xmin, xmax, ymin, ymax = domain
    max_grad = 0.0
    h = 1e-5
    points = [(xmin, ymin), (xmax, ymax), (xmin, ymax), (xmax, ymin)]
    n_root = int(math.sqrt(n)) + 1
    for i in range(1, n_root):
        for j in range(1, n_root):
            x = xmin + i * (xmax - xmin) / n_root
            y = ymin + j * (ymax - ymin) / n_root
            points.append((x, y))
    for x, y in points:
        f = safe_eval(func, x, y, threshold)
        fx = safe_eval(func, x + h, y, threshold)
        fy = safe_eval(func, x, y + h, threshold)
        if math.isnan(f) or math.isnan(fx) or math.isnan(fy):
            continue
        dfdx = (fx - f) / h
        dfdy = (fy - f) / h
        grad = math.sqrt(dfdx ** 2 + dfdy ** 2)
        if grad > max_grad:
            max_grad = grad
    return max_grad * safety if max_grad > 0 else 1.0


class Canvas:
    def __init__(self, xlim: tuple = (-5, 5), ylim: tuple = (-5, 5), resolution: real = 80, aspect_ratio: real = 3):
        self.xlim = xlim
        self.ylim = ylim
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.plane: list = []
        self.clear()

    def _n_cols(self):
        return max(1, round(self.resolution))

    def _n_rows(self):
        return max(1, round(self.resolution / self.aspect_ratio))

    def plot(self, function, marker: str = '.', use_color: bool = False, return_points: bool = False):
        if not isinstance(marker, str) or (len(marker) != 1 and (not use_color)):
            raise ValueError('The parameter marker must be one character')
        left, right = self.xlim
        bottom, top = self.ylim
        n_cols = self._n_cols()
        n_rows = self._n_rows()
        x = linspace(left, right, n_cols + 1)
        y = list(map(function, x))
        plane_height = len(self.plane)
        col_offset = 11
        coordinates = []
        for i, y_val in enumerate(y):
            row_idx = round((top - y_val) / (top - bottom) * n_rows)
            if 0 <= row_idx < plane_height:
                self.plane[row_idx][i + col_offset] = marker
            if return_points:
                coordinates.append((x[i], y_val))
        if return_points:
            return coordinates

    def plot_contour(self, function, threshold: real = 0, marker: str = '.', use_color: bool = False,
                     return_points: bool = False):
        if not isinstance(marker, str) or (len(marker) != 1 and (not use_color)):
            raise ValueError('The parameter marker must be one character')
        left, right = self.xlim
        bottom, top = self.ylim
        n_cols = self._n_cols()
        n_rows = self._n_rows()
        plane_height = len(self.plane)
        plane_width = len(self.plane[0]) if plane_height > 0 else 0
        col_offset = 11
        domain = (left, right, bottom, top)
        L = estimate_lipschitz(function, domain, threshold)
        cache = {}

        def cached_eval(x, y):
            key = (x, y)
            if key not in cache:
                cache[key] = safe_eval(function, x, y, threshold)
            return cache[key]

        max_depth = 20  # FIXME 调整细分最大次数
        min_cell_size = min((right - left) / n_cols, (top - bottom) / n_rows) * 0.2
        dx_pix = (right - left) / n_cols
        dy_pix = (top - bottom) / n_rows
        coordinates = []
        stack_main = [(left, right, bottom, top)]
        while stack_main:
            xmin, xmax, ymin, ymax = stack_main.pop()
            dx = xmax - xmin
            dy = ymax - ymin
            v_bl = cached_eval(xmin, ymin)
            v_br = cached_eval(xmax, ymin)
            v_tl = cached_eval(xmin, ymax)
            v_tr = cached_eval(xmax, ymax)
            vals = []
            has_nan = False
            for v in (v_bl, v_br, v_tl, v_tr):
                if math.isnan(v):
                    has_nan = True
                else:
                    vals.append(v)
            if not vals:
                continue
            if not has_nan:
                fmin = min(vals)
                fmax = max(vals)
                hd = 0.5 * math.sqrt(dx * dx + dy * dy)
                if not (fmin - L * hd <= 0.0 <= fmax + L * hd):
                    continue
                if max(dx, dy) <= min_cell_size:
                    has_pos = any(v > 1e-8 for v in vals)
                    has_neg = any(v < -1e-8 for v in vals)
                    is_near_zero = any(abs(v) <= 1e-8 for v in vals)
                    if (has_pos and has_neg) or is_near_zero:
                        cx = (xmin + xmax) / 2
                        cy = (ymin + ymax) / 2
                        row_idx = round((top - cy) / (top - bottom) * n_rows)
                        target_col = round((cx - left) / (right - left) * n_cols) + col_offset
                        if 0 <= row_idx < plane_height and 0 <= target_col < plane_width:
                            self.plane[row_idx][target_col] = marker
                            if return_points:
                                coordinates.append((cx, cy))
                    continue
                else:
                    xm = (xmin + xmax) / 2
                    ym = (ymin + ymax) / 2
                    stack_main.append((xmin, xm, ymin, ym))
                    stack_main.append((xm, xmax, ymin, ym))
                    stack_main.append((xmin, xm, ym, ymax))
                    stack_main.append((xm, xmax, ym, ymax))
            else:
                if dx <= dx_pix and dy <= dy_pix:
                    temp_stack = [(xmin, xmax, ymin, ymax, 0)]
                    stop_pixel = False
                    while temp_stack and not stop_pixel:
                        pxmin, pxmax, pymin, pymax, pdepth = temp_stack.pop()
                        pdx = pxmax - pxmin
                        pdy = pymax - pymin
                        pv_bl = cached_eval(pxmin, pymin)
                        pv_br = cached_eval(pxmax, pymin)
                        pv_tl = cached_eval(pxmin, pymax)
                        pv_tr = cached_eval(pxmax, pymax)
                        pvals = []
                        for v in (pv_bl, pv_br, pv_tl, pv_tr):
                            if not math.isnan(v):
                                pvals.append(v)
                        if not pvals:
                            continue
                        has_pos = any(v > 1e-8 for v in pvals)
                        has_neg = any(v < -1e-8 for v in pvals)
                        if not (has_pos and has_neg):
                            is_possible = False
                            if math.isnan(pv_bl) or math.isnan(pv_tl):
                                for y_val in (pymin, pymax):
                                    v1 = cached_eval(pxmax, y_val)
                                    v2 = cached_eval(pxmax + pdx, y_val)
                                    if not math.isnan(v1) and not math.isnan(v2) and abs(v2 - v1) > 1e-12:
                                        t = -v1 / (v2 - v1)
                                        if 0 <= t <= 1:
                                            pred_x = pxmax + t * pdx
                                            row_idx = round((top - y_val) / (top - bottom) * n_rows)
                                            target_col = round((pred_x - left) / (right - left) * n_cols) + col_offset
                                            if 0 <= row_idx < plane_height and 0 <= target_col < plane_width:
                                                self.plane[row_idx][target_col] = marker
                                                if return_points:
                                                    coordinates.append((pred_x, y_val))
                                            stop_pixel = True
                                            break
                                        elif t < 0:
                                            is_possible = True
                                            break
                            if stop_pixel:
                                continue
                            if not is_possible and (math.isnan(pv_br) or math.isnan(pv_tr)):
                                for y_val in (pymin, pymax):
                                    v1 = cached_eval(pxmin, y_val)
                                    v2 = cached_eval(pxmin - pdx, y_val)
                                    if not math.isnan(v1) and not math.isnan(v2) and abs(v2 - v1) > 1e-12:
                                        t = -v1 / (v2 - v1)
                                        if 0 <= t <= 1:
                                            pred_x = pxmin - t * pdx
                                            row_idx = round((top - y_val) / (top - bottom) * n_rows)
                                            target_col = round((pred_x - left) / (right - left) * n_cols) + col_offset
                                            if 0 <= row_idx < plane_height and 0 <= target_col < plane_width:
                                                self.plane[row_idx][target_col] = marker
                                                if return_points:
                                                    coordinates.append((pred_x, y_val))
                                            stop_pixel = True
                                            break
                                        elif t < 0:
                                            is_possible = True
                                            break
                            if stop_pixel:
                                continue
                            if not is_possible and (math.isnan(pv_bl) or math.isnan(pv_br)):
                                for x_val in (pxmin, pxmax):
                                    v1 = cached_eval(x_val, pymax)
                                    v2 = cached_eval(x_val, pymax + pdy)
                                    if not math.isnan(v1) and not math.isnan(v2) and abs(v2 - v1) > 1e-12:
                                        t = -v1 / (v2 - v1)
                                        if 0 <= t <= 1:
                                            pred_y = pymax + t * pdy
                                            row_idx = round((top - pred_y) / (top - bottom) * n_rows)
                                            target_col = round((x_val - left) / (right - left) * n_cols) + col_offset
                                            if 0 <= row_idx < plane_height and 0 <= target_col < plane_width:
                                                self.plane[row_idx][target_col] = marker
                                                if return_points:
                                                    coordinates.append((x_val, pred_y))
                                            stop_pixel = True
                                            break
                                        elif t < 0:
                                            is_possible = True
                                            break
                            if stop_pixel:
                                continue
                            if not is_possible and (math.isnan(pv_tl) or math.isnan(pv_tr)):
                                for x_val in (pxmin, pxmax):
                                    v1 = cached_eval(x_val, pymin)
                                    v2 = cached_eval(x_val, pymin - pdy)
                                    if not math.isnan(v1) and not math.isnan(v2) and abs(v2 - v1) > 1e-12:
                                        t = -v1 / (v2 - v1)
                                        if 0 <= t <= 1:
                                            pred_y = pymin - t * pdy
                                            row_idx = round((top - pred_y) / (top - bottom) * n_rows)
                                            target_col = round((x_val - left) / (right - left) * n_cols) + col_offset
                                            if 0 <= row_idx < plane_height and 0 <= target_col < plane_width:
                                                self.plane[row_idx][target_col] = marker
                                                if return_points:
                                                    coordinates.append((x_val, pred_y))
                                            stop_pixel = True
                                            break
                                        elif t < 0:
                                            is_possible = True
                                            break
                            if stop_pixel:
                                continue
                            if not is_possible and len(pvals) >= 2:
                                valid_corners = []
                                if not math.isnan(pv_bl): valid_corners.append((pxmin, pymin, pv_bl))
                                if not math.isnan(pv_br): valid_corners.append((pxmax, pymin, pv_br))
                                if not math.isnan(pv_tl): valid_corners.append((pxmin, pymax, pv_tl))
                                if not math.isnan(pv_tr): valid_corners.append((pxmax, pymax, pv_tr))
                                for i in range(len(valid_corners)):
                                    x0, y0, v0 = valid_corners[i]
                                    for j in range(i + 1, len(valid_corners)):
                                        x1, y1, v1 = valid_corners[j]
                                        diff_v = v1 - v0
                                        if abs(diff_v) < 1e-12: continue
                                        t = -v0 / diff_v
                                        pred_x = x0 + t * (x1 - x0)
                                        pred_y = y0 + t * (y1 - y0)
                                        if pxmin <= pred_x <= pxmax and pymin <= pred_y <= pymax:
                                            is_possible = True
                                            break
                                        if pred_x < pxmin and (math.isnan(pv_bl) or math.isnan(pv_tl)):
                                            is_possible = True
                                            break
                                        if pred_x > pxmax and (math.isnan(pv_br) or math.isnan(pv_tr)):
                                            is_possible = True
                                            break
                                        if pred_y < pymin and (math.isnan(pv_bl) or math.isnan(pv_br)):
                                            is_possible = True
                                            break
                                        if pred_y > pymax and (math.isnan(pv_tl) or math.isnan(pv_tr)):
                                            is_possible = True
                                            break
                                    if is_possible: break
                            if not is_possible:
                                continue
                            if pdepth >= max_depth:
                                cx = (pxmin + pxmax) / 2
                                cy = (pymin + pymax) / 2
                                has_pos = any(v > 1e-8 for v in pvals)
                                has_neg = any(v < -1e-8 for v in pvals)
                                is_near_zero = any(abs(v) <= 1e-8 for v in pvals)
                                if (has_pos and has_neg) or is_near_zero:
                                    row_idx = round((top - cy) / (top - bottom) * n_rows)
                                    target_col = round((cx - left) / (right - left) * n_cols) + col_offset
                                    if 0 <= row_idx < plane_height and 0 <= target_col < plane_width:
                                        self.plane[row_idx][target_col] = marker
                                        if return_points:
                                            coordinates.append((cx, cy))
                                stop_pixel = True
                                continue
                            pxm = (pxmin + pxmax) / 2
                            pym = (pymin + pymax) / 2
                            temp_stack.append((pxmin, pxm, pymin, pym, pdepth + 1))
                            temp_stack.append((pxm, pxmax, pymin, pym, pdepth + 1))
                            temp_stack.append((pxmin, pxm, pym, pymax, pdepth + 1))
                            temp_stack.append((pxm, pxmax, pym, pymax, pdepth + 1))
                else:
                    xm = (xmin + xmax) / 2
                    ym = (ymin + ymax) / 2
                    stack_main.append((xmin, xm, ymin, ym))
                    stack_main.append((xm, xmax, ymin, ym))
                    stack_main.append((xmin, xm, ym, ymax))
                    stack_main.append((xm, xmax, ym, ymax))
        if return_points:
            return coordinates

    def plot_complex(self, function, projection: str = 'ri', marker: str = '.', use_color: bool = False,
                     return_points: bool = False):
        if not isinstance(marker, str) or (len(marker) != 1 and (not use_color)):
            raise ValueError('The parameter marker must be one character')
        left, right = self.xlim
        bottom, top = self.ylim
        n_cols = self._n_cols()
        n_rows = self._n_rows()
        x = linspace(left, right, n_cols + 1)
        y = linspace(top, bottom, n_rows + 1)
        coordinates = [((c0, c1), function(complex(c0, c1))) for p1, c1 in enumerate(y) for p0, c0 in enumerate(x)]
        plane_height = len(self.plane)
        plane_width = len(self.plane[0]) - 11 if plane_height > 0 else 0
        col_offset = 11
        for px, py in coordinates:
            if projection == 'ri':
                _c0, _c1 = (py.real, py.imag)
            else:
                raise ValueError('Other modes are currently not supported')
            c0, c1 = (round((_c0 - left) / (right - left) * n_cols), round((top - _c1) / (top - bottom) * n_rows))
            if 0 <= c0 < plane_width and 0 <= c1 < plane_height:
                self.plane[c1][c0 + col_offset] = marker
        if return_points:
            return coordinates

    def clear(self):
        left, right = self.xlim
        bottom, top = self.ylim
        resolution, aspect_ratio = (self.resolution, self.aspect_ratio)
        if abs(aspect_ratio) != aspect_ratio:
            raise ValueError('The aspect_ratio cannot be less than zero')
        if resolution <= 0:
            raise ValueError('The resolution must be positive')
        if right - left <= 0 or top - bottom <= 0:
            raise ValueError('The defined width or height must be positive')
        n_cols = self._n_cols()
        n_rows = self._n_rows()
        x = linspace(left, right, n_cols + 1)
        mid_y = (top + bottom) / 2
        mid_x = (right + left) / 2
        y_mid_row = round(n_rows / 2)
        self.plane = [[' '] * 10 + ['|'] + [' '] * len(x) if _ != y_mid_row else [' '] * 10 + ['|'] + list(
            ' '.join('_' * (len(x) // 2 + 1))) for _ in range(n_rows)] + [
                         [' '] * 10 + ['|'] + ['_'] * len(x)]
        fmt = '{:.2e}'
        self.plane[0][:10] = fmt.format(top).rjust(10)
        self.plane[-1][:10] = fmt.format(bottom).rjust(10)
        self.plane[y_mid_row][:10] = fmt.format(mid_y).rjust(10)
        self.plane.append(
            [' '] * 11 + list(fmt.format(left).ljust(10)) + list(fmt.format(mid_x).center(len(x) - 20)) + list(
                fmt.format(right).rjust(10)))

    def render(self) -> str:
        return '\n'.join([''.join(row) for row in self.plane])

    def __str__(self):
        return self.render()
