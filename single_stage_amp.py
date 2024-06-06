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

    # NOTE, SPICE syntax for AC analysis:
    # .ac dec nd fstart fstop
    # https://ngspice.sourceforge.io/docs/ngspice-html-manual/manual.xhtml#subsec__AC__Small_Signal_AC
    # Same hints as in Leme, 2012:
    # ac dec 5 10 1g
    acCircuit.hints["ac"]["variation"] = "dec"
    acCircuit.hints["ac"]["points"] = 5
    acCircuit.hints["ac"]["start"] = 10
    acCircuit.hints["ac"]["end"] = 1e9

    # NOTE, SPICE syntax for TRAN analysis:
    # .tran tstep tstop
    # https://ngspice.sourceforge.io/docs/ngspice-html-manual/manual.xhtml#subsec__TRAN__Transient_Analysis
    # Leme, 2012:
    # tran 0.1us 25us
    tranCircuit.hints["tran"]["step"] = 0.1e-6
    tranCircuit.hints["tran"]["end"] = 25e-6

    # TODO: look for a more elegant way than one try-except
    # for each circuit metric.
    # NOTE: Don't use a single try-except, as a good
    # performance in any single variable is worth keeping.
    errors = set()

    # NOTE: gain should not throw
    # (it simply gets the first or the maximum amplitude value)
    try:
        gain = numpy.absolute(acCircuit.gain)
    except:
        gain = 0
        errors.add("gain")

    try:
        bandwidth = acCircuit.unityGainFrequency
    except:
        bandwidth = 0
        errors.add("unityGainFrequency")

    # NOTE: staticPower (i.e., OP analysis) should not throw
    try:
        power = acCircuit.staticPower
    except:
        power = numpy.inf
        errors.add("staticPower")

    # TODO: IDEIA atualizar Vpulse antes de análise transiente.
    try:
        # NOTE: hints defined above.
        slew_rate = tranCircuit.slewRate
    except:
        # TODO: Investigate.
        # NOTE: quite verbose.
        # print("slewRate undefined, hints:", tranCircuit.hints["tran"], end=", ")
        # print("netlist:", tranCircuit.netlist)

        slew_rate = 0
        errors.add("slewRate")

    # NOTE: area doesn't throw exceptions.
    area = area_key(individual)

    if len(errors) > 0:
        for name in errors:
            print(name, "undefined", end=", ")

        print("")

    return gain, bandwidth, power, slew_rate, area


def params_from_ind(individual):
    return dict(zip((IV + LS + WS), individual))


# Clean up to run script interactively (ipython).
classes = [
    "FitnessAmp",
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
