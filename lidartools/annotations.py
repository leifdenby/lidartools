import numpy as np
import datetime


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
