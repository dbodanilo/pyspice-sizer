import numpy
import pickle
import random

from datetime import datetime
from deap import algorithms, base, creator, tools
from math import factorial
from matplotlib import pyplot
from multiprocessing import Pool
from sizer import CircuitTemplate


# cm, l12, l34, l5, l6, l7, l8,
#     w12, w34, w5, w6, w7, w8.
IND_SIZE = 13

C_MIN, C_MAX = 1e-12, 10e-12
LW_MIN, LW_MAX = 3.5e-7, 3.5e-4

C = "cm"
PAIRS = ["12", "34", "5", "6", "7", "8"]
LS = ["l" + p for p in PAIRS]
WS = ["w" + p for p in PAIRS]

# NPOP, NGEN = 50, 50
# CXPB, MUTPB = 0.5, 0.2

# Multiobjective
# $A_{v0}$, $f_T$
NOBJ = 2
K = 10
NDIM = 1 + len(LS) + len(WS)
P = 12
# (NOBJ + P - 1)! / (P! * (NOBJ - 1)!)
H = factorial(NOBJ + P - 1) / factorial(P) / factorial(NOBJ - 1)
BOUND_LOW = [C_MIN] + [LW_MIN] * (len(LS) + len(WS))
BOUND_UP = [C_MAX] + [LW_MAX] * (len(LS) + len(WS))

MU = int(H + (4 - H % 4))
NGEN = 400
CXPB = 1.0
MUTPB = 1.0

EXTS = ["pdf", "png"]


# NOTE: area statistic.
def area_key(individual):
    a = 0.0
    for l, w in zip(individual[1:7], individual[7:]):
        a += l * w

    return a


def bandwidth_key(individual):
    params = params_from_ind(individual)
    circuit = circuitTemplate(params)

    try:
        return circuit.bandwidth
    # bandwidth undefined
    except:
        return 1


def bandwidth_loss(circuit):
    try:
        return (5e3 - circuit.bandwidth) / 5e3
    except:
        print("bandwidth undefined")
        return 1


def checkBounds(func):
    def wrapper(*args, **kargs):
        offspring = func(*args, **kargs)
        for child in offspring:
            if child[0] < C_MIN:
                child[0] = C_MIN
            elif child[0] > C_MAX:
                child[0] = C_MAX
            for i in range(1, len(child)):
                if child[i] < LW_MIN:
                    child[i] = LW_MIN
                elif child[i] > LW_MAX:
                    child[i] = LW_MAX
        return offspring
    return wrapper


def evaluate(individual):
    # start = time()
    # NOTE: Put it in a circuit.
    params = params_from_ind(individual)
    circuit = circuitTemplate(params)

    gain = numpy.absolute(circuit.gain)

    try:
        bandwidth = circuit.bandwidth
    except:
        bandwidth = 1

    # end = time()
    # print(f"total loss: {_loss:10.5f}, {end - start:5.4f}s per seed")
    return gain, bandwidth


def evaluate_to_snd(individual):
    return (individual, toolbox.evaluate(individual))


def gain_key(individual):
    params = params_from_ind(individual)
    circuit = circuitTemplate(params)

    return numpy.absolute(circuit.gain)


def gain_loss(circuit):
    return (1e3 - numpy.absolute(circuit.gain)) / 1e3


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


def params_from_ind(ind):
    return dict(zip((C, *LS, *WS), ind))


def to_snd(f):
    def wrapper(x):
        return (x, f(x))
    return wrapper


# Clean up to run script interactively (ipython).
classes = [
    "FitnessMax",
    "FitnessMin",
    "Individual",
]
for c in classes:
    if hasattr(creator, c):
        delattr(creator, c)

with open("./demos/two-stage-amplifier/two-stage-amp.cir") as f:
    circuitTemplate = CircuitTemplate(f.read())

ref_points = tools.uniform_reference_points(NOBJ, P)

# A_{v0}, f_T[, SR]
creator.create("FitnessMax", base.Fitness, weights=(1.0, ) * NOBJ)

# loss[, Pwr, Area]
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# A_{v0}, f_T
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
# toolbox.register("attr_c", random.uniform, C_MIN, C_MAX)
# toolbox.register("attr_lw", random.uniform, LW_MIN, LW_MAX)
# toolbox.register("attr_lws", tools.initRepeat, list,
#                  toolbox.attr_lw, n=(IND_SIZE - 1))
toolbox.register("individual", tools.initIterate, creator.Individual,
                 generator)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
# toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=30.0, indpb=1.0/NDIM)

# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)

# NOTE: No need to decorate Bounded function.
# toolbox.decorate("mate", checkBounds)
# toolbox.decorate("mutate", checkBounds)

# stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_area = tools.Statistics(key=area_key)
stats_gain = tools.Statistics(key=gain_key)
stats_bandwidth = tools.Statistics(key=bandwidth_key)

# fitness=stats_fit
mstats = tools.MultiStatistics(gain=stats_gain,
                               bandwidth=stats_bandwidth,
                               area=stats_area)

mstats.register("avg", numpy.mean, axis=0)
mstats.register("std", numpy.std, axis=0)
mstats.register("min", numpy.min, axis=0)
mstats.register("max", numpy.max, axis=0)

logbook = tools.Logbook()

# First group by the common `gen`, then group by chapters
# , "fitness"
logbook.header = "gen", "evals", "gain", "bandwidth", "area"
logbook.chapters["gain"].header = "avg", "std", "min", "max"
logbook.chapters["bandwidth"].header = "avg", "std", "min", "max"
logbook.chapters["area"].header = "avg", "std", "min", "max"
# logbook.chapters["fitness"].header = "avg", "std", "min", "max"


def main(seed=None):
    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    random.seed(seed)

    pop = toolbox.population(n=MU)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = mstats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
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

        # Compile statistics about the new population
        record = mstats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    prefix = f"./out/{_now}_deap_nsga3-"

    with open((prefix + "logbook.pickle"), "wb") as f:
        pickle.dump(logbook, f, protocol=pickle.DEFAULT_PROTOCOL)

    with open((prefix + "pop.pickle"), "wb") as f:
        pickle.dump(pop, f, protocol=pickle.DEFAULT_PROTOCOL)

    gen = logbook.select("gen")
    gain_avgs = logbook.chapters["gain"].select("avg")
    bandwidth_avgs = logbook.chapters["bandwidth"].select("avg")

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

    # Ideally, the graph would be increasing (/),
    # so loc="lower right", seems to be appropriate.
    fig.legend()

    for fname in ((prefix + "gain_bandwidth-yscale_log." + ext) for ext in EXTS):
        fig.savefig(fname)
    # pyplot.show()

    pop_fit = numpy.array([ind.fitness.values for ind in pop])

    # figsize=(7, 7)
    fig, ax = pyplot.subplots()
    ax.scatter(pop_fit[:, 0], pop_fit[:, 1], marker="o",
               s=24, label="Final Population")

    ax.scatter(ref_points[:, 0], ref_points[:, 1], marker="o",
               s=24, label="Reference Points")

    ax.set_xlabel("Gain (V/V)")
    ax.set_ylabel("Bandwidth (Hz)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.autoscale(tight=True)
    fig.legend()
    # pyplot.tight_layout()

    for fname in (prefix + "pareto_front-scale_log." + ext for ext in EXTS):
        fig.savefig(fname)
    pyplot.show()

    return pop, logbook


if __name__ == "__main__":
    pool = Pool()
    toolbox.register("map", pool.map)

    pop, logbook = main()
