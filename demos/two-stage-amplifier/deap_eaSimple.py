import numpy
import pickle
import random

from datetime import datetime
from deap import algorithms, base, creator, tools
from matplotlib import pyplot
from multiprocessing import Pool
from sizer import CircuitTemplate
from time import time


# cm, l12, l34, l5, l6, l7, l8,
#     w12, w34, w5, w6, w7, w8.
IND_SIZE = 13

C_MIN, C_MAX = 1e-12, 10e-12
LW_MIN, LW_MAX = 3.5e-7, 3.5e-4

C = "cm"
PAIRS = ["12", "34", "5", "6", "7", "8"]
LS = ["l" + p for p in PAIRS]
WS = ["w" + p for p in PAIRS]


# Clean up to run script interactively (ipython -i).
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


def gainLoss(circuit):
    return (1e3 - numpy.absolute(circuit.gain)) / 1e3


def bandwidthLoss(circuit):
    try:
        return (5e3 - circuit.bandwidth) / 5e3
    except:
        print("bandwidth undefined")
        return 1


def loss(circuit):
    gl = gainLoss(circuit)
    bl = bandwidthLoss(circuit)
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


def evaluate(individual):
    start = time()

    # NOTE: Put it in a circuit.
    params = params_from_ind(individual)
    circuit = circuitTemplate(params)
    _loss = loss(circuit)

    end = time()
    print(f"total loss: {_loss:10.5f}, {end - start:5.4f}s per seed")

    return (_loss,)


# A_{v0}, f_T, SR
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# loss, Pwr, Area
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# loss
creator.create("Individual", list, fitness=creator.FitnessMin)


def generator():
    return [toolbox.attr_c()] + toolbox.attr_lws()


toolbox = base.Toolbox()
toolbox.register("attr_c", random.uniform, C_MIN, C_MAX)
toolbox.register("attr_lw", random.uniform, LW_MIN, LW_MAX)
toolbox.register("attr_lws", tools.initRepeat, list,
                 toolbox.attr_lw, n=(IND_SIZE - 1))
toolbox.register("individual", tools.initIterate, creator.Individual,
                 generator)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


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


def gain_key(individual):
    params = params_from_ind(individual)
    circuit = circuitTemplate(params)

    return numpy.absolute(circuit.gain)


stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_area = tools.Statistics(key=area_key)
stats_gain = tools.Statistics(key=gain_key)
stats_bandwidth = tools.Statistics(key=bandwidth_key)

mstats = tools.MultiStatistics(gain=stats_gain, bandwidth=stats_bandwidth)

mstats.register("avg", numpy.mean, axis=0)
mstats.register("std", numpy.std, axis=0)
mstats.register("min", numpy.min, axis=0)
mstats.register("max", numpy.max, axis=0)

logbook = tools.Logbook()

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)


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


toolbox.decorate("mate", checkBounds)
toolbox.decorate("mutate", checkBounds)


def main():
    NPOP, NGEN = 50, 50
    CXPB, MUTPB = 0.5, 0.2

    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    pop = toolbox.population(n=NPOP)
    pop, logbook = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN,
                                       stats=mstats, verbose=True)

    # First group by the common `gen`, then group by chapters
    logbook.header = "gen", "gain", "bandwidth"
    logbook.chapters["gain"].header = "avg", "std"
    logbook.chapters["bandwidth"].header = "avg", "std"

    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    prefix = f"./out/{_now}_deap-eaSimple_"

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

    # Ideally, the graph would be increasing (/).
    fig.legend(loc="lower right")

    # pyplot.show()
    for fname in ((prefix + "gain-bandwidth-yscale_log" + ext) for ext in [".pdf", ".png"]):
        fig.savefig(fname)

    return pop, logbook


if __name__ == "__main__":
    pool = Pool()
    toolbox.register("map", pool.map)

    pop, logbook = main()
