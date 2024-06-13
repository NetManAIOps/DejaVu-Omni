import numpy as np
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "cohend",
    "moving_average_anomaly_detection",
    "t_test_anomaly_detection",
]

from scipy.stats import t


def cohend(d1: np.ndarray, d2: np.ndarray, axis: int = -1, debug: bool = False) -> np.ndarray:
    assert len(d1.shape) == len(d2.shape), f"{d1.shape=}, {d2.shape=}"
    if axis < 0:
        axis = len(d1.shape) + axis
    d1 = np.moveaxis(d1, axis, -1)
    d2 = np.moveaxis(d2, axis, -1)
    assert d1.shape[:-1] == d2.shape[:-1], f"{d1.shape=}, {d2.shape=} (after moving axis)"

    # calculate the size of samples
    n1, n2 = d1.shape[-1], d2.shape[-1]
    # calculate the variance of the samples
    s1, s2 = np.var(d1, axis=-1, ddof=1), np.var(d2, ddof=1, axis=-1)
    s1 = np.nan_to_num(s1, nan=1.)
    s2 = np.nan_to_num(s2, nan=1.)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    if debug:
        logger.debug(f"{n1=} {n2=} {s1=} {s2=} {s=} {u1=} {u2=}")
    # calculate the effect size
    return (u1 - u2) / s


def moving_average_anomaly_detection(
        kpi: np.ndarray, axis: int = -1, upper_limit: float = 0.97, lower_limit: float = 0.03,
        window_size: int = 60, min_std: float = 0.1, debug: bool = False
) -> np.ndarray:
    """
    :param kpi:
    :param axis:
    :param upper_limit:
    :param lower_limit:
    :param window_size:
    :param min_std:
    :param debug:
    :return: The anomaly status for the last window (-1, 0, +1)
    """
    pi = 3.1415926
    if axis < 0:
        axis = np.ndim(kpi) + axis
    kpi = np.moveaxis(kpi, axis, -1)

    x_windows = sliding_window_view(kpi, window_shape=window_size, axis=-1)
    x = x_windows[..., -1]
    mean = np.mean(x_windows[..., :-1], axis=-1)
    std = np.clip(np.std(x_windows[..., :-1], axis=-1), a_min=min_std, a_max=None)
    cdf = np.arctan((x - mean) / std) / pi + 0.5
    if debug:
        logger.debug(f"\n{x=} \n{mean=} \n{std=} \n{cdf=}")
    ret = np.zeros_like(x)
    ret[cdf > upper_limit] = 1
    ret[cdf < lower_limit] = -1
    ret = np.concatenate([
        np.zeros(kpi.shape[:-1] + (window_size - 1,), dtype=ret.dtype),
        ret,
    ], axis=-1)
    assert ret.shape == kpi.shape

    ret = np.moveaxis(ret, -1, axis)
    return ret


def t_test_anomaly_detection(
        kpi: np.ndarray, window_size: int = 10,
        axis: int = -1, significance_level: float = 0.05,
        min_std: float = 1e-1,
        debug: bool = False,
) -> np.ndarray:
    """
    :param kpi:
    :param window_size:
    :param axis:
    :param significance_level:
    :param min_std:
    :param debug:
    :return: Whether the final window changes significantly
    """
    if axis < 0:
        axis = np.ndim(kpi) + axis
    data1 = np.take(kpi, indices=np.arange(0, kpi.shape[axis] - window_size), axis=axis)
    data2 = np.take(kpi, indices=np.arange(kpi.shape[axis] - window_size, kpi.shape[axis]), axis=axis)
    if debug:
        logger.debug(f"{data1=} {data2=}")
    l1 = data1.shape[axis]
    l2 = data2.shape[axis]
    if debug:
        logger.debug(f"{l1=} {l2=}")
    df = l1 + l2 - 2
    # robust
    # mean1 = th.median(data1, dim=dim).values
    # std1 = th.median(th.abs(data1 - mean1[:, None]), dim=dim).values
    # mean2 = th.median(data2, dim=dim).values
    # std2 = th.median(th.abs(data2 - mean2[:, None]), dim=dim).values
    # original
    mean1 = np.mean(data1, axis=axis)
    std1 = np.clip(np.std(data1, axis=axis, ddof=1), a_min=min_std, a_max=None)
    mean2 = np.mean(data2, axis=axis)
    std2 = np.clip(np.std(data2, axis=axis, ddof=1), a_min=min_std, a_max=None)
    sed = np.sqrt(
        ((l1 - 1) * np.square(std1) + (l2 - 1) * np.square(std2)) / df * (1 / l1 + 1 / l2)
    )
    t_stat = (mean1 - mean2) / sed
    p = np.array((1.0 - t.cdf(np.abs(t_stat), df)) * 2.0)
    if debug:
        logger.debug(f"{mean1=} {std1=} {mean2=} {std2=} {sed=} {df=} {t_stat=} {p=}")
    return (p < significance_level).astype(np.float) * np.sign(t_stat)
