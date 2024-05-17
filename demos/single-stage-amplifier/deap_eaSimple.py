import numpy
import os
import pickle
import random

from datetime import datetime
from deap import algorithms, base, creator, tools
from matplotlib import pyplot
from multiprocessing import Pool
from sizer import CircuitTemplate


_PLOT = True

# ipol, vpol, l12, l34, l56, l78, l9, l10
#             w12, w34, w56, w78, w9, w10
IND_SIZE = 14

# 0,1\micro\ampere, 100\micro\ampere
I_MIN, I_MAX = 0.1e-6, 100e-6

# 0,5\micro\meter, 100\micro\meter
LW_MIN, LW_MAX = 0.5e-6, 100e-6

# 0,7\volt, 2,3\volt
V_MIN, V_MAX = 0.7, 2.3

IV = ["ipol", "vpol"]
PAIRS = ["12", "34", "56", "78", "9", "10"]
LS = ["l" + p for p in PAIRS]
WS = ["w" + p for p in PAIRS]

BOUND_LOW = [I_MIN, V_MIN] + [LW_MIN] * (len(LS) + len(WS))
BOUND_UP = [I_MAX, V_MAX] + [LW_MAX] * (len(LS) + len(WS))

NDIM = 2 + len(LS) + len(WS)

# DEAP:
NGEN = 50

# Leme, 2012:
# NGEN = 1000  # or 6000
NPOP = 200
CXPB = 0.8
MUTPB = 0.07
ETA_C = 20.0
ETA_M = 12.0

EXTS = ["pdf", "png"]


# NOTE: area statistic.
def area_key(individual):
    a = 0.0
    ls = individual[2:(2 + len(LS))]
    ws = individual[-len(WS):]
    assert len(ls) == len(ws)
    for l, w in zip(ls, ws):
        a += l * w

    return a


# TODO: avoid keys that instantiate a new circuit.
def bandwidth_key(individual):
    params = params_from_ind(individual)
    circuit = circuitTemplate(params)

    try:
        return circuit.unityGainFrequency
    # bandwidth undefined
    except:
        return 0


def bandwidth_loss(circuit):
    try:
        # Leme, 2012: median(f_T) [50%] = 397.747 MHz
        return (400e6 - circuit.unityGainFrequency) / 400e6
    except:
        return 1


def evaluate(individual):
    params = params_from_ind(individual)
    circuit = circuitTemplate(params)
    _loss = loss(circuit)

    return (_loss,)


# TODO: avoid instantiating a new circuit.
def gain_key(individual):
    params = params_from_ind(individual)
    circuit = circuitTemplate(params)

    return numpy.absolute(circuit.gain)


def gain_loss(circuit):
    # Leme, 2012: median(A_{v0}) [50%] = 38.4639 dB [83.7905 V/V]
    return (100 - numpy.absolute(circuit.gain)) / 100


def generator():
    return [random.uniform(low, up) for low, up in zip(BOUND_LOW, BOUND_UP)]


def loss(circuit):
    gl = gain_loss(circuit)
    bl = bandwidth_loss(circuit)
    gls = numpy.sign(gl)
    bls = numpy.sign(bl)

    # Only one has reached the goal, not the other.
    if gls != bls:
        if gls == -1:
            # Zero the reached goal, to avoid biasing the
            # optimization towards it.
            gl = 0
        elif bls == -1:
            bl = 0

    return numpy.sum((gl, bl))


def params_from_ind(individual):
    return dict(zip((IV + LS + WS), individual))


# TODO (Leme, 2012): use phase margin as a restriction.
def phase_margin_loss(circuit):
    try:
        return numpy.maximum(0, (60 - circuit.phaseMargin) / 60) ** 2
        # return np.maximum(0, (60 - circuit.phaseMargin) / 60)
    except:
        return 0


# Clean up to run script interactively (ipython).
classes = [
    "FitnessMax",
    "FitnessMin",
    "Individual",
]
for c in classes:
    if hasattr(creator, c):
        delattr(creator, c)

with open("./demos/single-stage-amplifier/single-stage-amp.cir") as f:
    circuitTemplate = CircuitTemplate(f.read())


# A_{v0}, f_T, SR
creator.create("FitnessMax", base.Fitness, weights=(1.0,) * 3)

# loss[, Pwr, Area]
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# loss
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual,
                 generator)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)

# Leme, 2012:
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=ETA_C)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=ETA_M, indpb=1.0/NDIM)

toolbox.register("select", tools.selTournament, tournsize=3)

stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)

# TODO: avoid _keys, as they take time to instantiate the circuit.
stats_gain = tools.Statistics(key=gain_key)
stats_bandwidth = tools.Statistics(key=bandwidth_key)

mstats = tools.MultiStatistics(fitness=stats_fit,
                               gain=stats_gain,
                               bandwidth=stats_bandwidth)
mstats.register("avg", numpy.mean, axis=0)
mstats.register("std", numpy.std, axis=0)
mstats.register("min", numpy.min, axis=0)
mstats.register("max", numpy.max, axis=0)


def main(seed=None):
    # YYYY-mm-dd_HH-mm
    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    random.seed(seed)

    pop = toolbox.population(n=NPOP)
    pop, logbook = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN,
                                       stats=mstats, verbose=True)

    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    prefix = f"./out/single-stage-amp/{_now}_deap_eaSimple-"
    prefix_dir = prefix[:prefix.rfind("/")]
    os.makedirs(prefix_dir, exist_ok=True)

    # First group by the common `gen`, `nevals`, then group by chapters
    logbook.header = "gen", "nevals", "gain", "bandwidth", "fitness"
    logbook.chapters["gain"].header = "avg", "std", "min", "max"
    logbook.chapters["bandwidth"].header = "avg", "std", "min", "max"
    logbook.chapters["fitness"].header = "avg", "std", "min", "max"

    with open((prefix + "logbook.pickle"), "wb") as f:
        pickle.dump(logbook, f, protocol=pickle.DEFAULT_PROTOCOL)

    with open((prefix + "pop.pickle"), "wb") as f:
        pickle.dump(pop, f, protocol=pickle.DEFAULT_PROTOCOL)

    gen = logbook.select("gen")
    gain_avgs = logbook.chapters["gain"].select("avg")
    bandwidth_avgs = logbook.chapters["bandwidth"].select("avg")

    pop_fit = numpy.array([ind.fitness.values for ind in pop])
    pop_keys = numpy.array(
        [(gain_key(ind), bandwidth_key(ind)) for ind in pop])

    ind_min_loss = pop[numpy.argmin(pop_fit)]
    circuit = circuitTemplate(params_from_ind(ind_min_loss))

    print(circuit.netlist)
    print("total loss:", loss(circuit))
    print("optimal parameters:", circuit.parameters)

    # TODO: reconsider printing those, as they might fail.
    # print("bandwidth:", circuit.bandwidth)
    # print("gain:", circuit.gain)
    # print("phase margin:", circuit.phaseMargin)

    if _PLOT:
        fig, ax1 = pyplot.subplots()

        ax1.plot(gen, gain_avgs, "b-", label="Average Gain")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Gain (V/V)", color="b")
        ax1.set_yscale("log")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()
        ax2.plot(gen, bandwidth_avgs, "r-", label="Average Bandwidth")
        ax2.set_ylabel("Bandwidth (Hz)", color="r")
        ax2.set_yscale("log")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        fig.legend()

        for fname in ((prefix + "gain_bandwidth-yscale_log." + ext) for ext in EXTS):
            fig.savefig(fname)
        # NOTE: savefig() before show(), as the latter clears the figure.
        pyplot.show()

        fig, ax = pyplot.subplots()

        # Same order as in Bode plot (frequency on x axis).
        ax.scatter(pop_keys[:, 1], pop_keys[:, 0], marker="o",
                   s=24, label="Final Population")

        ax.set_xlabel("Bandwidth (Hz)")
        ax.set_ylabel("Gain (V/V)")

        ax.set_xscale("log")
        ax.set_yscale("log")

        fig.legend()

        for fname in ((prefix + "pop-scale_log." + ext) for ext in EXTS):
            fig.savefig(fname)
        pyplot.show()

    return pop, logbook


if __name__ == "__main__":
    pool = Pool()
    toolbox.register("map", pool.map)

    pop, logbook = main()
