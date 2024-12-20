import matplotlib.pyplot as plt
import numpy as np

import logging
import sizer
import sys

from datetime import datetime


sys.path.append(".")
logger = logging.getLogger()
# logger.setLevel(0)

_PLOT = False

with open("./demos/single-stage-amplifier/single-stage-amp.cir") as f:
    circuitTemplate = sizer.CircuitTemplate(f.read())


def bandwidthLoss(circuit):
    try:
        return np.maximum(0, (5e+3 - circuit.unityGainFrequency) / 5e+3) ** 2
    except:
        print("unityGainFrequency undefined")
        return 1


def gainLoss(circuit):
    return np.maximum(0, (1e+3 - np.abs(circuit.gain)) / 1e+3) ** 2
    # return np.maximum(0, (1e+3 - np.abs(circuit.gain)) / 1e+3)
    # return (1e+3 - np.abs(circuit.gain)) / 1e+3


def phaseMarginLoss(circuit):
    try:
        return np.maximum(0, (60 - circuit.phaseMargin) / 60) ** 2
        # return np.maximum(0, (60 - circuit.phaseMargin) / 60)
    except:
        return 0


def loss(circuit):
    return np.sum([phaseMarginLoss(circuit), gainLoss(circuit), bandwidthLoss(circuit)])


bounds = {
    w: [0.5e-6, 100e-6] for w in ["w12", "w34", "w56", "w78", "w9", "w10"]
}

bounds.update({
    l: [0.5e-6, 100e-6] for l in ["l12", "l34", "l56", "l78", "l9", "l10"]
})

bounds.update({
    a: [0.5e-6, 100e-6] for a in ["a12", "a34", "a56", "a78", "a9", "a10"]
})

bounds.update({
    p: [0.5e-6, 100e-6] for p in ["p12", "p34", "p56", "p78", "p9", "p10"]
})

bounds.update({
    nr: [0.5e-6, 100e-6] for nr in ["nr12", "nr34", "nr56", "nr78", "nr9", "nr10"]
})

optimizer = sizer.optimizers.Optimizer(
    circuitTemplate, loss, bounds, earlyStopLoss=0
)
# optimizer = sizer.optimizers.ScipyMinimizeOptimizer(circuitTemplate, loss, bounds, earlyStopLoss=0)
# circuit = circuitTemplate([bounds[i][0] for i in circuitTemplate.parameters])
# frequencies, frequencyResponse = circuit.getFrequencyResponse()
# raise Exception()

# YYYY-mm-dd_HH-mm
_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
print(_now)

circuit = optimizer.run()
print(circuit.netlist)
print("total loss:", loss(circuit))
print("optimal parameters",
      dict(zip(circuitTemplate.parameters, circuit.parameters)))
print("bandwidth:", circuit.unityGainFrequency)
print("gain:", circuit.gain)
print("phase margin:", circuit.phaseMargin)

_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
print(_now)


if _PLOT:
    plt.rcParams["axes.grid"] = True

    frequencies, frequencyResponse = circuit.getFrequencyResponse()

    plt.subplot(211)
    plt.plot(frequencies, np.abs(frequencyResponse))
    plt.xscale("log")
    plt.yscale("log")
    plt.vlines(sizer.calculators.unityGainFrequency(
        frequencies, frequencyResponse), 0, 1e+3)

    plt.subplot(212)
    phaseResponse = np.angle(frequencyResponse, deg=True)
    phaseResponse[np.where(phaseResponse > 0)] -= 360
    plt.plot(frequencies, phaseResponse)
    plt.xscale("log")
    plt.vlines(sizer.calculators.unityGainFrequency(
        frequencies, frequencyResponse), -180, 0)
    plt.show()
