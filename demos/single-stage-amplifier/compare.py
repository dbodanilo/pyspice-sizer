import numpy
import os
import pandas
import pickle
import sys

from datetime import datetime
from deap import tools
from deap.tools._hypervolume import hv
from deap.tools.emo import (
    assignCrowdingDist,
    associate_to_niche,
    find_extreme_points,
    find_intercepts,
    sortLogNondominated,
)
from math import comb, log10
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from single_stage_amp import creator, evaluate, NOBJ


# Deb, 2014
P = 6
# (P + NOBJ - 1)! / (P! * (NOBJ - 1)!)
H = comb(P + NOBJ - 1, P)  # P = 6, NOBJ = 5 -> H = 210
# first multiple of 4 higher than or equal to H.
MU = int(H + (4 - H % 4))  # H = 210 -> MU = 212

ref_points = tools.uniform_reference_points(NOBJ, P)

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


def assignNichingDist(pop):
    pareto_fronts = sortLogNondominated(pop, MU)

    fits_weighted = numpy.array(
        [ind.fitness.wvalues for f in pareto_fronts for ind in f])
    fits_weighted *= -1

    best_point = numpy.min(fits_weighted, axis=0)
    worst_point = numpy.max(fits_weighted, axis=0)

    extreme_points = find_extreme_points(fits_weighted, best_point)
    front_worst = numpy.max(
        fits_weighted[:sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = find_intercepts(
        extreme_points, best_point, worst_point, front_worst)
    _, dist = associate_to_niche(
        fits_weighted, ref_points, best_point, intercepts)

    for i, d in enumerate(dist):
        pop[i].fitness.niching_dist = d


# dB = 20 log_{10}(ratio)
def db_from_ratio(r):
    return 20 * log10(r)


# ratio = 10 ^ {dB / 20}
def ratio_from_db(db):
    return 10 ** (db / 20)


def evaluate_series(ind):
    return pandas.Series(evaluate(ind), index=targets)


# seed_leme <- [1241..1245]
def main(seed_leme=1241, prefix_dir="./out/single-stage-amp/", script="compare-metrics_leme", deap_path="deap_nsga3/2024-05-22_15-42_deap_nsga3-params_deb-seed_1241-pop.pickle"):
    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now, f"pop={deap_path}")

    # TODO: update script details as needed, e.g., metrics.
    script = f"{script}-seed_{seed_leme}"
    prefix = f"{prefix_dir}compare/{_now}_{script}-"

    # TODO: make cleaner distinction between nsga3's and compare's directories.
    os.makedirs(prefix_dir + "compare/", exist_ok=True)

    deap_path = prefix_dir + deap_path
    with open(deap_path, "rb") as pop_file:
        deap_pop = pickle.load(pop_file)

    # pickled pop already has fitness values, but evaluate() has
    # been updated.
    deap_fits = numpy.array(list(map(evaluate, deap_pop)))

    # TODO: actually handle the cause for inf values in Pwr.
    numpy.place(deap_fits, ~numpy.isfinite(deap_fits), [1])

    for ind, fit in zip(deap_pop, deap_fits):
        ind.fitness.values = fit

    assignCrowdingDist(deap_pop)
    deap_cds = numpy.array([ind.fitness.crowding_dist for ind in deap_pop])

    assignNichingDist(deap_pop)
    deap_nds = numpy.array([ind.fitness.niching_dist for ind in deap_pop])

    with open((prefix + "deap_pop.pickle"), "wb") as f:
        pickle.dump(deap_pop, f, protocol=pickle.DEFAULT_PROTOCOL)

    Y_deap = pandas.DataFrame(deap_fits, columns=targets)

    Y_deap_weighted = Y_deap * y_weights

    ref_deap = numpy.max(Y_deap_weighted, axis=0) + 1

    deap_scaler = MinMaxScaler()

    Y_deap_scaled_deap = pandas.DataFrame(
        deap_scaler.fit_transform(Y_deap),
        columns=targets
    )
    Y_deap_scaled_deap_weighted = Y_deap_scaled_deap * y_weights

    ref_deap_scaled_deap = numpy.max(Y_deap_scaled_deap_weighted, axis=0) + 1

    X_train = X[X["Semente"] == seed_leme].drop(columns=["N.", "Semente"])
    Y_train = Y[Y["Semente"] == seed_leme].drop(columns=["N.", "Semente"])

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

    Y_train_scaled_deap = pandas.DataFrame(
        deap_scaler.transform(Y_train),
        columns=targets
    )
    Y_train_scaled_deap_weighted = Y_train_scaled_deap * y_weights

    Y_deap_scaled_leme = pandas.DataFrame(
        leme_scaler.transform(Y_deap),
        columns=targets
    )
    Y_deap_scaled_leme_weighted = Y_deap_scaled_leme * y_weights

    # same values as in X_train.
    train_pop = X_train.apply(creator.Individual, axis=1)
    sim_pop = train_pop.copy()

    # same values as in Y_train.
    train_fits = Y_train.to_numpy()

    train_pop = train_pop.to_numpy()
    for ind, fit in zip(train_pop, train_fits):
        ind.fitness.values = fit

    assignCrowdingDist(train_pop)
    train_cds = numpy.array([ind.fitness.crowding_dist for ind in train_pop])

    assignNichingDist(train_pop)
    train_nds = numpy.array([ind.fitness.niching_dist for ind in train_pop])

    with open((prefix + "train_pop.pickle"), "wb") as f:
        pickle.dump(train_pop, f, protocol=pickle.DEFAULT_PROTOCOL)

    # same values as in Y_sim.
    sim_fits = sim_pop.apply(evaluate_series).to_numpy()

    sim_pop = sim_pop.to_numpy()
    for ind, fit in zip(sim_pop, sim_fits):
        ind.fitness.values = fit

    assignCrowdingDist(sim_pop)
    sim_cds = numpy.array([ind.fitness.crowding_dist for ind in sim_pop])

    assignNichingDist(sim_pop)
    sim_nds = numpy.array([ind.fitness.niching_dist for ind in sim_pop])

    with open((prefix + "sim_pop.pickle"), "wb") as f:
        pickle.dump(sim_pop, f, protocol=pickle.DEFAULT_PROTOCOL)

    Y_sim = pandas.DataFrame(sim_fits, columns=targets)

    Y_sim_weighted = pandas.DataFrame(
        numpy.array([ind.fitness.wvalues for ind in sim_pop]) * -1,
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

    Y_sim_scaled_deap = pandas.DataFrame(
        deap_scaler.transform(Y_sim),
        columns=targets
    )
    Y_sim_scaled_deap_weighted = Y_sim_scaled_deap * y_weights

    Y_sim_scaled_leme = pandas.DataFrame(
        leme_scaler.transform(Y_sim),
        columns=targets
    )

    Y_sim_scaled_leme_weighted = Y_sim_scaled_leme * y_weights

    Y_deap_scaled_sim = pandas.DataFrame(
        sim_scaler.transform(Y_deap),
        columns=targets
    )
    Y_deap_scaled_sim_weighted = Y_deap_scaled_sim * y_weights

    Y_train_scaled_sim = pandas.DataFrame(
        sim_scaler.transform(Y_train),
        columns=targets
    )
    Y_train_scaled_sim_weighted = Y_train_scaled_sim * y_weights

    print("\npop:", deap_path)

    # NOTE: print accurate information.
    print("\nsimulation score:", end="")
    print("\n-----------------")
    print("[python calc]")
    print("raw:")
    print(Y_sim_r2)
    print("avg:", numpy.mean(Y_sim_r2))

    print("\nHypervolume (hv, max):", end="")
    print("\n----------------------")

    # TODO: reevaluate the relevance of raw hypervolume.
    print("raw hv:")
    print("deap:", hv.hypervolume(Y_deap_weighted.to_numpy(), ref_deap.to_numpy()))
    print("leme:", hv.hypervolume(Y_train_weighted.to_numpy(), ref_leme.to_numpy()))
    print("simulated:", hv.hypervolume(
        Y_sim_weighted.to_numpy(), ref_sim.to_numpy()
    ))

    # TODO: reevaluate the relevance of scaling the
    # hypervolume to the deap result.
    print("\ndeap-scaled hv:")
    print("deap:", hv.hypervolume(
        Y_deap_scaled_deap_weighted.to_numpy(),
        ref_deap_scaled_deap.to_numpy()
    ))
    print("leme:", hv.hypervolume(
        Y_train_scaled_deap_weighted.to_numpy(),
        ref_deap_scaled_deap.to_numpy()
    ))
    print("simulated:", hv.hypervolume(
        Y_sim_scaled_deap_weighted.to_numpy(),
        ref_deap_scaled_deap.to_numpy()
    ))

    print("\nleme-scaled hv:")
    print("deap:", hv.hypervolume(
        Y_deap_scaled_leme_weighted.to_numpy(),
        ref_leme_scaled_leme.to_numpy()
    ))
    print("leme:", hv.hypervolume(
        Y_train_scaled_leme_weighted.to_numpy(),
        ref_leme_scaled_leme.to_numpy()
    ))
    print("simulated:", hv.hypervolume(
        Y_sim_scaled_leme_weighted.to_numpy(),
        ref_leme_scaled_leme.to_numpy()
    ))

    print("\nsim-scaled hv:")
    print("deap:", hv.hypervolume(
        Y_deap_scaled_sim_weighted.to_numpy(),
        ref_sim_scaled_sim.to_numpy()
    ))
    print("leme:", hv.hypervolume(
        Y_train_scaled_sim_weighted.to_numpy(),
        ref_sim_scaled_sim.to_numpy()
    ))
    print("simulated:", hv.hypervolume(
        Y_sim_scaled_sim_weighted.to_numpy(),
        ref_sim_scaled_sim.to_numpy()
    ))

    print("\nCrowding Distance (cd, max):", end="")
    print("\n----------------------------")

    print("\nmean cd (finite):")
    print("deap:", numpy.mean(deap_cds[numpy.isfinite(deap_cds)]))
    print("leme:", numpy.mean(train_cds[numpy.isfinite(train_cds)]))
    print("simulated:", numpy.mean(sim_cds[numpy.isfinite(sim_cds)]))

    print("\nmedian cd (finite):")
    print("deap:", numpy.median(deap_cds[numpy.isfinite(deap_cds)]))
    print("leme:", numpy.median(train_cds[numpy.isfinite(train_cds)]))
    print("simulated:", numpy.median(sim_cds[numpy.isfinite(sim_cds)]))

    print("\nmedian cd (all):")
    print("deap:", numpy.median(deap_cds))
    print("leme:", numpy.median(train_cds))
    print("simulated:", numpy.median(sim_cds))

    print("\nNiching Distance (nd, min):", end="")
    print("\n----------------------------")

    print("\nmean nd:")
    print("deap:", numpy.mean(deap_nds))
    print("leme:", numpy.mean(train_nds))
    print("simulated:", numpy.mean(sim_nds))

    print("\nmedian nd:")
    print("deap:", numpy.median(deap_nds))
    print("leme:", numpy.median(train_nds))
    print("simulated:", numpy.median(sim_nds))

    # TODO: plot some view of the Pareto frontiers as well.

    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)

    return train_pop, sim_pop, deap_pop


if __name__ == "__main__":
    stdout = sys.stdout

    prefix_dir = "./out/single-stage-amp/"
    os.makedirs(prefix_dir + "compare/", exist_ok=True)

    script = f"compare-metrics_leme-crowding_dist-niching_dist"

    deap_paths = [
        "deap_nsga3/2024-06-07_01-11_deap_nsga3-params_deb-seed_1241-pop.pickle",
        "deap_nsga3/2024-06-07_02-39_deap_nsga3-params_deb-seed_1242-pop.pickle",
        "deap_nsga3/2024-06-07_04-10_deap_nsga3-params_deb-seed_1243-pop.pickle",
        "deap_nsga3/2024-06-07_05-33_deap_nsga3-params_deb-seed_1244-pop.pickle",
        "deap_nsga3/2024-06-07_06-57_deap_nsga3-params_deb-seed_1245-pop.pickle",
    ]

    leme_seeds = range(1241, 1246)

    seeds_paths = list(zip(leme_seeds, deap_paths))

    # NOTE: index 0 <- leme_seed 1241
    #       index 1 <- leme_seed 1242
    #       ...
    for seed, path in seeds_paths:
        _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        print(_now, "pop:", path)

        prefix = f"{prefix_dir}compare/{_now}_{script}-seed_{seed}-"

        with open((prefix + "run.log"), "a") as run_log:
            sys.stdout = run_log

            # Make them global, so they're available for interactive use.
            train_pop, sim_pop, deap_pop = main(
                seed_leme=seed, prefix_dir=prefix_dir, script=script, deap_path=path)

            sys.stdout = stdout

        _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        print(_now, "pop:", path)
