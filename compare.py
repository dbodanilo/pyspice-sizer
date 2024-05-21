import numpy
import pandas
import pickle

from datetime import datetime
from deap.tools._hypervolume import hv
from math import log10
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from single_stage_amp import creator, evaluate


# dB = 20 log_{10}(ratio)
def db_from_ratio(r):
    return 20 * log10(r)


# ratio = 10 ^ {dB / 20}
def ratio_from_db(db):
    return 10 ** (db / 20)


seed_leme = 1241
timestamp = "2024-05-17_10-55"
model = "deap_nsga3-params_deb"
seed_deap = seed_leme
gen = None

_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
print(_now)

leme_data = pandas.read_csv("leme-fronteira.csv", delimiter="\t")

Y = leme_data[["N.", "Semente", "A_{v0}", "f_{T}", "Pwr", "SR", "Area"]]

# set difference from Y's columns, except "N." and "Semente".
X = leme_data.drop(columns=Y.columns.drop(["N.", "Semente"]))

X_train = X[X["Semente"] == seed_leme].drop(columns=["N.", "Semente"])
Y_train = Y[Y["Semente"] == seed_leme].drop(columns=["N.", "Semente"])

# I_{pol}: \micro\ampere;
# V_{pol}: \volt;
# Ws and Ls: \micro\meter.
x_prefixes = (1e-6, 1.0) + (1e-6,) * 12
x_prefixes = pandas.Series(x_prefixes, index=X_train.columns)

# A_{v0}: \deci\bel (deal with it later);
# f_{T}: \mega\hertz;
# Pwr: \milli\watt
# SR: \volt\per\micro\second;
# Area: \micro\meter\squared.
y_prefixes = (1.0, 1e6, 1e-3, 1e6, 1e-12)
y_prefixes = pandas.Series(y_prefixes, index=Y_train.columns)

X_train *= x_prefixes
Y_train *= y_prefixes

# \deci\bel
Y_train["A_{v0}"] = Y_train["A_{v0}"].apply(ratio_from_db)

# minimization as default (weight=1.0)
y_weights = (-1.0, -1.0, 1.0, -1.0, 1.0)
y_weights = pandas.Series(y_weights, index=Y_train.columns)

Y_train_weighted = Y_train * y_weights

ref_leme = numpy.max(Y_train_weighted, axis=0) + 1

leme_scaler = MinMaxScaler()

Y_train_scaled_leme = pandas.DataFrame(
    leme_scaler.fit_transform(Y_train), columns=Y_train.columns)

Y_train_scaled_leme_weighted = Y_train_scaled_leme * y_weights

ref_leme_scaled_leme = numpy.max(Y_train_scaled_leme_weighted, axis=0) + 1

prefix = f"./out/single-stage-amp/{timestamp}_{model}-seed_{seed_deap}-"
fname = prefix + "pop" + ("" if gen is None else f"-gen_{gen}") + ".pickle"
with open(fname, "rb") as pop_file:
    pop = pickle.load(pop_file)

fitnesses_old = pandas.DataFrame(
    [ind.fitness.values for ind in pop],
    columns=Y_train.columns
)

# pickled pop already has fitness values, but evaluate() has
# been updated.
fitnesses = list(map(evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

X_pop = X_train.apply(creator.Individual, axis=1)
Y_sim = list(map(evaluate, X_pop))

for ind, fit in zip(X_pop, Y_sim):
    ind.fitness.values = fit

Y_sim = pandas.DataFrame(Y_sim, columns=Y_train.columns)

obj = pandas.DataFrame(
    fitnesses,
    columns=Y_train.columns
)
wobj = pandas.DataFrame(
    numpy.array([ind.fitness.wvalues for ind in pop]) * -1,
    columns=Y_train.columns
)

Y_sim_weighted = pandas.DataFrame(
    numpy.array([ind.fitness.wvalues for ind in X_pop]) * -1,
    columns=Y_train.columns
)

Y_sim_r2 = pandas.Series(
    r2_score(Y_train, Y_sim, multioutput="raw_values"),
    index=Y_train.columns
)

ref_deap = numpy.max(wobj, axis=0) + 1
ref_leme_sim = numpy.max(Y_sim_weighted, axis=0) + 1

obj_scaled_leme = pandas.DataFrame(
    leme_scaler.transform(obj),
    columns=Y_train.columns
)

Y_sim_scaled_leme = pandas.DataFrame(
    leme_scaler.transform(Y_sim),
    columns=Y_train.columns
)

wobj_scaled_leme = obj_scaled_leme * y_weights

Y_sim_scaled_leme_weighted = Y_sim_scaled_leme * y_weights

deap_scaler = MinMaxScaler()

obj_scaled_deap = pandas.DataFrame(
    deap_scaler.fit_transform(obj),
    columns=Y_train.columns
)

Y_train_scaled_deap = pandas.DataFrame(
    deap_scaler.transform(Y_train),
    columns=Y_train.columns
)

Y_sim_scaled_deap = pandas.DataFrame(
    deap_scaler.transform(Y_sim),
    columns=Y_train.columns
)

wobj_scaled_deap = obj_scaled_deap * y_weights

Y_train_scaled_deap_weighted = Y_train_scaled_deap * y_weights

Y_sim_scaled_deap_weighted = Y_sim_scaled_deap * y_weights

ref_deap_scaled = numpy.max(wobj_scaled_deap, axis=0) + 1


def main():
    print("deap:", timestamp, model, seed_deap, gen, "leme:", seed_leme)

    # TODO: reevaluate the relevance of raw hypervolume.
    print("raw hv:")
    print("leme:", hv.hypervolume(Y_train_weighted.to_numpy(), ref_leme.to_numpy()))
    print("deap:", hv.hypervolume(wobj.to_numpy(), ref_deap.to_numpy()))
    print("simulated:", hv.hypervolume(
        Y_sim_weighted.to_numpy(), ref_leme_sim.to_numpy()
    ))

    print("\nleme-scaled hv:")
    print("leme:", hv.hypervolume(
        Y_train_scaled_leme_weighted.to_numpy(),
        ref_leme_scaled_leme.to_numpy()
    ))
    print("deap:", hv.hypervolume(
        wobj_scaled_leme.to_numpy(),
        ref_leme_scaled_leme.to_numpy()
    ))
    print("simulated:", hv.hypervolume(
        Y_sim_scaled_leme_weighted.to_numpy(),
        ref_leme_scaled_leme.to_numpy()
    ))

    # TODO: reevaluate the relevance of scaling the
    # hypervolume to the deap result.
    print("\ndeap-scaled hv:")
    print("leme:", hv.hypervolume(
        Y_train_scaled_deap_weighted.to_numpy(),
        ref_deap_scaled.to_numpy()
    ))
    print("deap:", hv.hypervolume(
        wobj_scaled_deap.to_numpy(),
        ref_deap_scaled.to_numpy()
    ))
    print("simulated:", hv.hypervolume(
        Y_sim_scaled_deap_weighted.to_numpy(),
        ref_deap_scaled.to_numpy()
    ))

    if seed_deap == seed_leme:
        print("simulation score:")
        print("raw:")
        print(Y_sim_r2)
        print("avg:", numpy.mean(Y_sim_r2))

    _now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(_now)


if __name__ == "__main__":
    main()
