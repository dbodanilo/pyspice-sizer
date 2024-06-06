import numpy
import os
import pandas
import pickle
import sys

from datetime import datetime
from deap.tools._hypervolume import hv
from math import log10
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from single_stage_amp import creator, evaluate


_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
print(_now)

leme_data = pandas.read_csv("leme-fronteira.csv", delimiter="\t")

Y = leme_data[["N.", "Semente", "A_{v0}", "f_{T}", "Pwr", "SR", "Area"]]

# set difference from Y's columns, except "N." and "Semente".
X = leme_data.drop(columns=Y.columns.drop(["N.", "Semente"]))

descriptors = X.columns.drop(["N.", "Semente"])
targets = Y.columns.drop(["N.", "Semente"])

# I_{pol}: \micro\ampere;
# V_{pol}: \volt;
# Ws and Ls: \micro\meter.
x_prefixes = (1e-6, 1.0) + (1e-6,) * 12
x_prefixes = pandas.Series(x_prefixes, index=descriptors)

# A_{v0}: \deci\bel (deal with it later);
# f_{T}: \mega\hertz;
# Pwr: \milli\watt
# SR: \volt\per\micro\second;
# Area: \micro\meter\squared.
y_prefixes = (1.0, 1e6, 1e-3, 1e6, 1e-12)
y_prefixes = pandas.Series(y_prefixes, index=targets)

# minimization as default (weight=1.0)
y_weights = (-1.0, -1.0, 1.0, -1.0, 1.0)
y_weights = pandas.Series(y_weights, index=targets)


# dB = 20 log_{10}(ratio)
def db_from_ratio(r):
    return 20 * log10(r)


# ratio = 10 ^ {dB / 20}
def ratio_from_db(db):
    return 10 ** (db / 20)


# seed <- [1241..1245]
def main(seed=1241, prefix_dir="./out/single-stage-amp/", script="compare-metrics_leme"):
    # seed_leme = 1242
    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now, f"seed={seed}")

    os.makedirs(prefix_dir, exist_ok=True)
    # TODO: update script details as needed, e.g., metrics.
    script = f"{script}-seed_{seed}"
    prefix = f"{prefix_dir}{_now}_{script}-"

    X_train = X[X["Semente"] == seed].drop(columns=["N.", "Semente"])
    Y_train = Y[Y["Semente"] == seed].drop(columns=["N.", "Semente"])

    X_train *= x_prefixes
    Y_train *= y_prefixes

    # \deci\bel
    Y_train["A_{v0}"] = Y_train["A_{v0}"].apply(ratio_from_db)

    Y_train_weighted = Y_train * y_weights

    ref_leme = numpy.max(Y_train_weighted, axis=0) + 1

    leme_scaler = MinMaxScaler()

    Y_train_scaled_leme = pandas.DataFrame(
        leme_scaler.fit_transform(Y_train), columns=targets)

    Y_train_scaled_leme_weighted = Y_train_scaled_leme * y_weights

    ref_leme_scaled_leme = numpy.max(Y_train_scaled_leme_weighted, axis=0) + 1

    X_pop = X_train.apply(creator.Individual, axis=1)
    Y_sim = list(map(evaluate, X_pop))

    for ind, fit in zip(X_pop, Y_sim):
        ind.fitness.values = fit

    with open((prefix + "pop.pickle"), "wb") as f:
        pickle.dump(X_pop, f, protocol=pickle.DEFAULT_PROTOCOL)

    Y_sim = pandas.DataFrame(Y_sim, columns=targets)

    Y_sim_weighted = pandas.DataFrame(
        numpy.array([ind.fitness.wvalues for ind in X_pop]) * -1,
        columns=targets
    )

    ref_sim = numpy.max(Y_sim_weighted, axis=0) + 1

    sim_scaler = MinMaxScaler()

    Y_sim_scaled_sim = pandas.DataFrame(
        sim_scaler.fit_transform(Y_sim), columns=targets)

    Y_sim_scaled_sim_weighted = Y_sim_scaled_sim * y_weights

    ref_sim_scaled_sim = numpy.max(Y_sim_scaled_sim_weighted, axis=0) + 1

    Y_sim_r2 = pandas.Series(
        r2_score(Y_train, Y_sim, multioutput="raw_values"),
        index=targets
    )

    Y_sim_scaled_leme = pandas.DataFrame(
        leme_scaler.transform(Y_sim),
        columns=targets
    )

    Y_sim_scaled_leme_weighted = Y_sim_scaled_leme * y_weights

    Y_train_scaled_sim = pandas.DataFrame(
        sim_scaler.transform(Y_train),
        columns=targets
    )

    Y_train_scaled_sim_weighted = Y_train_scaled_sim * y_weights

    # print("deap:", timestamp, model, seed_deap, gen, "leme:", seed_leme)
    print("\nseed:", seed)

    # TODO: reevaluate the relevance of raw hypervolume.
    print("raw hv:")
    print("leme:", hv.hypervolume(Y_train_weighted.to_numpy(), ref_leme.to_numpy()))
    print("simulated:", hv.hypervolume(
        Y_sim_weighted.to_numpy(), ref_sim.to_numpy()
    ))

    print("\nleme-scaled hv:")
    print("leme:", hv.hypervolume(
        Y_train_scaled_leme_weighted.to_numpy(),
        ref_leme_scaled_leme.to_numpy()
    ))
    print("simulated:", hv.hypervolume(
        Y_sim_scaled_leme_weighted.to_numpy(),
        ref_leme_scaled_leme.to_numpy()
    ))

    print("\nsim-scaled hv:")
    print("leme:", hv.hypervolume(
        Y_train_scaled_sim_weighted.to_numpy(),
        ref_sim_scaled_sim.to_numpy()
    ))
    print("simulated:", hv.hypervolume(
        Y_sim_scaled_sim_weighted.to_numpy(),
        ref_sim_scaled_sim.to_numpy()
    ))

    # if seed_deap == seed_leme:
    if True:
        # NOTE: print accurate information.
        print("[python calc]")
        print("simulation score:")
        print("raw:")
        print(Y_sim_r2)
        print("avg:", numpy.mean(Y_sim_r2))

    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    return X_train, X_pop, Y_train, Y_sim


if __name__ == "__main__":
    stdout = sys.stdout

    prefix_dir = "./out/single-stage-amp/"
    os.makedirs(prefix_dir, exist_ok=True)

    script = f"compare-metrics_leme"

    for seed in range(1241, 1246):
        _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        print(_now, "seed:", seed)

        prefix = f"{prefix_dir}{_now}_{script}-seed_{seed}-"

        with open((prefix + "run.log"), "a") as run_log:
            sys.stdout = run_log

            # Make them global, so they're available for interactive use.
            X_train, X_pop, Y_train, Y_sim = main(
                seed=seed, prefix_dir=prefix_dir, script=script)

            sys.stdout = stdout

        _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        print(_now, "seed:", seed)
