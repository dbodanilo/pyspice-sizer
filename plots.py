import numpy

from datetime import datetime
from matplotlib import pyplot


EXTS = ["pdf", "png"]


def plotMetric(df, ylabel):
    m_top = 1.1 * df.max(axis=None)

    fig, ax = pyplot.subplots()

    df.boxplot(ax=ax, grid=True)

    ax.set_ylabel(ylabel)

    ax.set_ylim((0, m_top))

    return fig


def plotSlewRate(time, wave, script="compare", prefix=None):
    if prefix is None:
        prefix = datetime.now().strftime("%Y-%m-%d_%H-%M")

    prefix = f"{prefix}_{script}-"

    w_lim = 1.1 * numpy.max(wave)

    fig, ax = pyplot.subplots()
    ax.plot(time, wave)

    # plot y from 0 to w_lim, to avoid scaling the output.
    ax.set_ylim((0, w_lim))

    for fname in (prefix + "time_wave." + ext for ext in EXTS):
        fig.savefig(fname)

    pyplot.show()
