import numpy
import pickle
import random

from datetime import datetime
from deap import algorithms, base, creator, tools
from matplotlib import pyplot
from multiprocessing import Pool
from sizer import CircuitTemplate, calculators
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
    return numpy.maximum(0, (1e3 - numpy.absolute(circuit.gain)) / 1e3) ** 2


def bandwidthLoss(circuit):
    try:
        return numpy.maximum(0, (5e3 - circuit.bandwidth) / 5e3) ** 2
    except:
        print("bandwidth undefined")
        return 1


def loss(circuit):
    return numpy.sum([gainLoss(circuit), bandwidthLoss(circuit)])


def evaluate(individual):
    start = time()

    # NOTE: Put it in a circuit.
    params = dict(zip((C, *LS, *WS), individual))
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
def area(individual):
    a = 0.0
    for l, w in zip(individual[1:7], individual[7:]):
        a += l * w

    return a


stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_area = tools.Statistics(key=area)

mstats = tools.MultiStatistics(fitness=stats_fit, area=stats_area)

mstats.register("avg", numpy.mean, axis=0)
mstats.register("std", numpy.std, axis=0)
mstats.register("min", numpy.min, axis=0)
mstats.register("max", numpy.max, axis=0)

logbook = tools.Logbook()
# First group by the common `gen`, then group by chapters
logbook.header = "gen", "fitness", "area"
logbook.chapters["fitness"].header = "avg", "std"
logbook.chapters["size"].header = "avg", "std"

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

    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    with open(f"./out/{_now}_deap-eaSimple_logbook.pickle", "wb") as f:
        pickle.dump(logbook, f, protocol=pickle.DEFAULT_PROTOCOL)

    gen = logbook.select("gen")
    fit_mins = logbook.chapters["fitness"].select("min")
    area_avgs = logbook.chapters["area"].select("avg")

    fig, ax1 = pyplot.subplots()
    ax2 = ax1.twinx()

    ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2.plot(gen, area_avgs, "r-", label="Average Area")
    ax2.set_ylabel("Area", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    fig.legend(loc="lower left")

    # pyplot.show()
    for fname in (f"./out/{_now}_deap_optimizers-fitness_area{ext}" for ext in [".pdf", ".png"]):
        fig.savefig(fname)

    return pop, logbook


if __name__ == "__main__":
    pool = Pool()
    toolbox.register("map", pool.map)

    main()
