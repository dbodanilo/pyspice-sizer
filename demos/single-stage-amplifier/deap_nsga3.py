import numpy
import os
import pickle
import random
import sys

from datetime import datetime
from deap import algorithms, base, creator, tools
from math import comb
from matplotlib import pyplot
from multiprocessing import Pool
from sizer import CircuitTemplate


_PLOT = True
_SHOW = False

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
# NGEN = 50  # eaSimple
# NGEN = 400  # nsga3

# Leme, 2012:
# NGEN = 1000  # or 6000
# NPOP = 200  # replaced by MU
# CXPB = 0.8
# MUTPB = 0.07
# ETA_C = 20.0
# ETA_M = 12.0

# Deb, 2014:
NGEN = 350  # [350..1000]
CXPB = 1.0  # p_c
MUTPB = 1.0  # TODO: review if this is correct.
P_M = 1.0/NDIM
ETA_C = 30.0
ETA_M = 20.0


# Multiobjective
# A_{v0}, f_T, Pwr, SR, Area
NOBJ = 5

# Deb, 2014
P = 6
# (P + NOBJ - 1)! / (P! * (NOBJ - 1)!)
H = comb(P + NOBJ - 1, P)  # P = 6, NOBJ = 5 -> H = 210

# first multiple of 4 higher than or equal to H.
MU = int(H + (4 - H % 4))  # H = 210 -> MU = 212

EXTS = ["pdf", "png"]


# NOTE: area statistic.
def area_key(individual):
    a = 0.0

    ls = individual[2:(2 + len(LS))]
    ws = individual[-len(WS):]
    assert len(ls) == len(ws)

    # l_pairs = ls[:-2]
    l_singles = ls[-2:]

    # w_pairs = ws[:-2]
    w_singles = ws[-2:]

    for l, w in zip(ls, ws):
        a += 2 * l * w

    # reiterate over singles, as there are only two of them.
    for l, w in zip(l_singles, w_singles):
        a -= l * w

    return a


# TODO: avoid keys that instantiate a new circuit.
def bandwidth_key(individual):
    params = params_from_ind(individual)
    circuit = circuitTemplate(params)

    try:
        return circuit.bandwidth
    # bandwidth undefined
    except:
        return 0


def bandwidth_loss(circuit):
    try:
        # Leme, 2012: median(f_T) [50%] = 397.747 MHz
        return (400e6 - circuit.bandwidth) / 400e6
    except:
        return 1


def evaluate(individual):
    params = params_from_ind(individual)
    circuit = circuitTemplate(params)

    # TODO: look for a more elegant way than one try-except
    # for each circuit metric.
    # NOTE: Don't use a single try-except, as a good
    # performance in any single variable is worth keeping.
    try:
        gain = numpy.absolute(circuit.gain)
    except:
        gain = 0

    try:
        bandwidth = circuit.bandwidth
    except:
        bandwidth = 0

    try:
        power = circuit.staticPower
    except:
        power = numpy.inf

    try:
        # TODO: evaluate the need for hints.
        slew_rate = circuit.slewRate
    except:
        slew_rate = 0

    # NOTE: area doesn't throw exceptions.
    area = area_key(individual)

    return gain, bandwidth, power, slew_rate, area


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
    "FitnessAmp",
    "FitnessMax",
    "FitnessMin",
    "Individual",
]
for c in classes:
    if hasattr(creator, c):
        delattr(creator, c)

with open("./demos/single-stage-amplifier/single-stage-amp.cir") as f:
    circuitTemplate = CircuitTemplate(f.read())

ref_points = tools.uniform_reference_points(NOBJ, P)

# A_{v0}, f_T, Pwr, SR, Area
creator.create("FitnessAmp", base.Fitness, weights=(1.0, 1.0, -1.0, 1.0, -1.0))

# A_{v0}, f_T, SR
creator.create("FitnessMax", base.Fitness, weights=(1.0, ) * 3)

# Pwr, Area[, loss]
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * 2)

# Five amplifier metrics
creator.create("Individual", list, fitness=creator.FitnessAmp)


toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual,
                 generator)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)

# Leme, 2012:
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=ETA_C)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=ETA_M, indpb=P_M)

toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)

# NOTE: avoid _keys, as they take time to instantiate the circuit.
# stats_gain = tools.Statistics(key=gain_key)
# stats_bandwidth = tools.Statistics(key=bandwidth_key)

mstats = tools.MultiStatistics(fitness=stats_fit)
mstats.register("avg", numpy.mean, axis=0)
mstats.register("std", numpy.std, axis=0)
mstats.register("min", numpy.min, axis=0)
mstats.register("max", numpy.max, axis=0)

# For testing, logbook and pop become global variables.
logbook = tools.Logbook()

# First group by the common `gen`, `nevals`, then group by chapters
logbook.header = "gen", "nevals", "fitness"
logbook.chapters["fitness"].header = "avg", "std", "min", "max"

pop = toolbox.population(n=MU)

checkpoint = 50


def main(seed=None, prefix_dir="./out/single-stage-amp/", model="deap_nsga3"):
    # YYYY-mm-dd_HH-mm
    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    os.makedirs(prefix_dir, exist_ok=True)
    model = f"{model}-seed_{seed}"
    prefix = f"{prefix_dir}{_now}_{model}-"

    random.seed(seed)

    # TODO: implement checkpoints, to unpause ongoing optimization.
    # def main(seed=None, gen=0):  # ...
    pop = toolbox.population(n=MU)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    logbook = tools.Logbook()

    # First group by the common `gen`, `nevals`, then group by chapters
    logbook.header = "gen", "nevals", "fitness"
    logbook.chapters["fitness"].header = "avg", "std", "min", "max"

    record = mstats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # 50, 100, ..., 900, 950.
        if gen % checkpoint == 0:
            with open((prefix + f"pop-gen_{gen}.pickle"), "wb") as f:
                pickle.dump(pop, f, protocol=pickle.DEFAULT_PROTOCOL)

        # Compile statistics about the new population
        record = mstats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        if gen % checkpoint == 0:
            with open((prefix + f"logbook-gen_{gen}.pickle"), "wb") as f:
                pickle.dump(logbook, f, protocol=pickle.DEFAULT_PROTOCOL)

    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    prefix = f"{prefix_dir}{_now}_{model}-"
    os.makedirs(prefix_dir, exist_ok=True)

    with open((prefix + "logbook.pickle"), "wb") as f:
        pickle.dump(logbook, f, protocol=pickle.DEFAULT_PROTOCOL)

    with open((prefix + "pop.pickle"), "wb") as f:
        pickle.dump(pop, f, protocol=pickle.DEFAULT_PROTOCOL)

    gen = logbook.select("gen")
    fitness_avgs = numpy.array(logbook.chapters["fitness"].select("avg"))
    gain_avgs = fitness_avgs[:, 0]
    bandwidth_avgs = fitness_avgs[:, 1]

    pop_fit = numpy.array([ind.fitness.values for ind in pop])

    # TODO: Evaluate the equivalent of a best individual for
    # multiobjective optimization.
    # ind_min_loss = pop[numpy.argmin(pop_fit)]
    # circuit = circuitTemplate(params_from_ind(ind_min_loss))

    # print(circuit.netlist)
    # print("total loss:", loss(circuit))
    # print("optimal parameters:", circuit.parameters)

    # TODO: reconsider printing those, as they might fail.
    # print("bandwidth:", circuit.bandwidth)
    # print("gain:", circuit.gain)
    # print("phase margin:", circuit.phaseMargin)

    # if _PLOT:
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
    if _SHOW:
        pyplot.show()
    else:
        pyplot.close("all")

    fig, ax = pyplot.subplots()

    # Same order as in Bode plot (frequency on x axis).
    ax.scatter(pop_fit[:, 1], pop_fit[:, 0], marker="o",
               s=24, label="Final Population")

    ax.scatter(ref_points[:, 1], ref_points[:, 0], marker="o",
               s=24, label="Reference Points")

    ax.set_xlabel("Bandwidth (Hz)")
    ax.set_ylabel("Gain (V/V)")

    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.legend()

    for fname in ((prefix + "pareto_front-scale_log." + ext) for ext in EXTS):
        fig.savefig(fname)

    if _SHOW:
        pyplot.show()
    else:
        pyplot.close("all")

    return pop, logbook


if __name__ == "__main__":
    pool = Pool()
    toolbox.register("map", pool.map)

    stdout = sys.stdout
    stderr = sys.stderr

    prefix_dir = "./out/single-stage-amp/"
    os.makedirs(prefix_dir, exist_ok=True)
    model = "deap_nsga3-params_deb"

    # pop, logbook = main()
    for seed in range(1241, 1246):
        _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        print(_now, "seed:", seed)
        prefix = f"{prefix_dir}{_now}_{model}-seed_{seed}-"

        with open((prefix + "run.log"), "a") as run_log:
            sys.stdout = run_log
            with open((prefix + "error.log"), "a") as err_log:
                sys.stderr = err_log
                main(seed=seed, prefix_dir=prefix_dir, model=model)
                sys.stderr = stderr
            sys.stdout = stdout

        _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        print(_now, "seed:", seed)
