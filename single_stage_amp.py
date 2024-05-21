import numpy

from deap import base, creator
from sizer import CircuitTemplate, CircuitTemplateList


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

# Multiobjective
# A_{v0}, f_T, Pwr, SR, Area
NOBJ = 5


# NOTE: area statistic.
def area_key(individual):
    a = 0.0

    ls = individual[2:(2 + len(LS))]
    ws = individual[-len(WS):]
    assert len(ls) == len(ws)

    for l, w in zip(ls, ws):
        a += 2 * l * w

    # l_pairs = ls[:-2]
    l_singles = ls[-2:]

    # w_pairs = ws[:-2]
    w_singles = ws[-2:]

    # reiterate over singles, as there are only two of them.
    for l, w in zip(l_singles, w_singles):
        a -= l * w

    return a


def evaluate(individual):
    params = params_from_ind(individual)
    circuits = circuitTemplateList(params)
    acCircuit = circuits[0]
    tranCircuit = circuits[1]

    # TODO: look for a more elegant way than one try-except
    # for each circuit metric.
    # NOTE: Don't use a single try-except, as a good
    # performance in any single variable is worth keeping.
    try:
        gain = numpy.absolute(acCircuit.gain)
    except:
        gain = 0

    try:
        # bandwidth = circuit.bandwidth
        bandwidth = acCircuit.unityGainFrequency
    except:
        bandwidth = 0

    try:
        power = acCircuit.staticPower
    except:
        power = numpy.inf

    try:
        # TODO: evaluate the need for hints.
        slew_rate = tranCircuit.slewRate
    except:
        slew_rate = 0

    # NOTE: area doesn't throw exceptions.
    area = area_key(individual)

    return gain, bandwidth, power, slew_rate, area


def params_from_ind(individual):
    return dict(zip((IV + LS + WS), individual))


# Clean up to run script interactively (ipython).
classes = [
    "FitnessAmp",
    # "FitnessMax",
    # "FitnessMin",
    "Individual",
]
for c in classes:
    if hasattr(creator, c):
        delattr(creator, c)

with open("./demos/single-stage-amplifier/single-stage-amp-ac.cir") as f:
    acTemplate = CircuitTemplate(f.read())

with open("./demos/single-stage-amplifier/single-stage-amp-tran.cir") as f:
    tranTemplate = CircuitTemplate(f.read())

circuitTemplateList = CircuitTemplateList((acTemplate, tranTemplate))

# A_{v0}, f_T, Pwr, SR, Area
creator.create("FitnessAmp", base.Fitness, weights=(1.0, 1.0, -1.0, 1.0, -1.0))

# Five amplifier metrics
creator.create("Individual", list, fitness=creator.FitnessAmp)
