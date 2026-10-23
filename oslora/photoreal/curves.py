import torch


class Curve:
    """Tone curve through control points, like the point curve of a photo editor. Monotone cubic (Fritsch-Carlson)
    interpolation: smooth, never overshoots between points and never reverses, so a curve built from sensible points
    cannot band or posterize. Points are (input, output) pairs in [0, 1]; (0, 0) and (1, 1) are added if missing."""

    def __init__(self, points=()):
        pts = dict(points)
        pts.setdefault(0.0, 0.0)
        pts.setdefault(1.0, 1.0)
        self.points = tuple(sorted(pts.items()))
        xs, ys = zip(*self.points)
        if any(b < a for a, b in zip(ys, ys[1:])):
            raise ValueError(f"curve points must not decrease: {self.points}")
        self._xs, self._ys = xs, ys
        self._slopes = self._fritsch_carlson(xs, ys)

    @staticmethod
    def _fritsch_carlson(xs, ys):
        n = len(xs)
        secants = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)]
        m = [secants[0]] + [(secants[i - 1] + secants[i]) / 2 for i in range(1, n - 1)] + [secants[-1]]
        for i, s in enumerate(secants):
            if s == 0:
                m[i] = m[i + 1] = 0.0
                continue
            a, b = m[i] / s, m[i + 1] / s
            if a * a + b * b > 9:  # keeps the segment monotone
                t = 3 / (a * a + b * b) ** 0.5
                m[i], m[i + 1] = t * a * s, t * b * s
        return m

    @property
    def is_identity(self):
        return all(x == y for x, y in self.points)

    def __call__(self, x):
        if self.is_identity:
            return x
        xs, ys, ms = (x.new_tensor(v) for v in (self._xs, self._ys, self._slopes))
        x = x.clamp(0, 1)
        i = torch.bucketize(x, xs[1:-1])
        x0, x1, y0, y1, m0, m1 = xs[i], xs[i + 1], ys[i], ys[i + 1], ms[i], ms[i + 1]
        h = x1 - x0
        t = (x - x0) / h
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * m0
                + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * m1)

    def slope_range(self, samples=2001):
        """(min, max) slope of the curve, for checking that a preset cannot band."""
        x = torch.linspace(0, 1, samples, dtype=torch.float64)
        s = torch.diff(self(x)) / torch.diff(x)
        return s.min().item(), s.max().item()
