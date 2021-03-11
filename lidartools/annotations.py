import numpy as np
import datetime
from matplotlib.dates import num2date, date2num


def get_sunrise_and_sunset_times_utc(t_start, t_end):
    """
    Return all sunrise and sunset times between `t_start` and `t_end` at BCO as
    `(time, kind)` where `time` will be a `np.datetim64` and kind is either
    `"sunrise"` or `"sunset"`
    # sunrise and sunset (local time): 0630, 1750
    # sunrise and sunset (UTC):        1030, 2150
    """

    def npdt64_to_dt(t):
        # NOTE: have to convert to microseconds here, otherwise we don't get a
        # datetime
        return t.astype("<M8[us]").astype(datetime.datetime)

    dt_start = npdt64_to_dt(t_start)
    dt_end = npdt64_to_dt(t_end)
    dt_current = datetime.datetime(
        year=dt_start.year, month=dt_start.month, day=dt_start.day
    )

    while dt_current < dt_end:
        dt_sunrise = dt_current + datetime.timedelta(hours=10, minutes=30)
        dt_sunset = dt_current + datetime.timedelta(hours=21, minutes=50)

        if dt_current < dt_sunrise < dt_end:
            yield (np.datetime64(dt_sunrise), "sunrise")
        if dt_current < dt_sunset < dt_end:
            yield (np.datetime64(dt_sunset), "sunset")
        dt_current += datetime.timedelta(days=1)


def annotate_with_sunrise_and_sunset(ax, da_time):
    """
    Annotate axes `ax` x-axis with all sunset and sunrise times in the time
    interval spanned by `da_time`
    """
    for (t, kind) in get_sunrise_and_sunset_times_utc(
        da_time.min().values, da_time.max().values
    ):
        ax.axvline(t, color="red", linestyle="--")
        ylim = ax.get_ylim()
        text = ax.text(
            t + np.timedelta64(2, "m"),
            0.8 * ylim[1],
            kind,
            color="red",
            fontsize=14,
            ha="center",
        )
        text.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="grey"))


@np.vectorize
def _to_dt64(x_):
    return np.datetime64(num2date(x_))


@np.vectorize
def _to_dt64ms(t):
    return np.timedelta64(t, "ms")


def add_approx_distance_axis(ax, t0, posn=1.2, v=10.0, units="m"):
    """
    Add extra x-axis to axes `ax` at position `posn` (can either be a float
    representing the position in figure coordinates, or "top" or "bottom") with
    constant velocity `v` in units `units` using reference time `t0` (expected
    to a `np.datetime64`)
    """
    if units == "m":
        s = 1.0
    elif units == "km":
        s = 1000.0
    else:
        raise NotImplementedError(units)

    def time_to_distance(x):
        x_dt = _to_dt64(x)
        t_offset = x_dt - t0

        x_dist = t_offset.astype("timedelta64[s]").astype(int) * v
        return x_dist / s

    def distance_to_time(x_dist):
        # I don't know why the F this function is ever called with an empty
        # array of values, but I have to check for it otherwise the vectorize
        # calls below fail
        if len(x_dist) == 0:
            return x_dist

        t_offset = x_dist * s / v  # [s]

        t = t0 + _to_dt64ms((t_offset * 1000).astype(int))
        return date2num(t)

    ax2 = ax.secondary_xaxis(posn, functions=(time_to_distance, distance_to_time))
    ax2.set_xlabel(f"approximate distance [{units}]")
    return ax2
