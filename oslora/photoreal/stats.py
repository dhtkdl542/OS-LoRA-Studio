SAMPLE_PIXELS = 1 << 18  # pixels sampled for statistics; sorting a full 3 MP frame dominated CPU time


def sample(t):
    """Evenly strided subset of t's values, at most about SAMPLE_PIXELS."""
    t = t.flatten()
    return t[:: max(1, t.numel() // SAMPLE_PIXELS)]


def percentiles(t, qs):
    return sample(t).quantile(t.new_tensor(qs)).tolist()

