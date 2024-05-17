import numpy
import pandas
import pickle

from deap.tools._hypervolume import hv
from math import log10
from sklearn.preprocessing import MinMaxScaler


# dB = 20 log_{10}(ratio)
def db_from_ratio(r):
    return 20 * log10(r)


# ratio = 10 ^ {dB / 20}
def ratio_from_db(db):
    return 10 ** (db / 20)


seed_leme = 1241
timestamp = "2024-05-17_10-55"
model = "deap_nsga3-params_deb"
seed_deap = 1241
gen = None

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

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train_scaled_self = pandas.DataFrame(
    x_scaler.fit_transform(X_train), columns=X_train.columns)
Y_train_scaled_self = pandas.DataFrame(
    y_scaler.fit_transform(Y_train), columns=Y_train.columns)

Y_train_scaled_self_weighted = Y_train_scaled_self * y_weights

ref_leme_scaled = numpy.max(Y_train_scaled_self_weighted, axis=0) + 1

prefix = f"./out/single-stage-amp/{timestamp}_{model}-seed_{seed_deap}-"
fname = prefix + "pop" + ("" if gen is None else f"-gen_{gen}") + ".pickle"
with open(fname, "rb") as pop_file:
    pop = pickle.load(pop_file)

obj = pandas.DataFrame(
    [ind.fitness.values for ind in pop], columns=Y_train.columns)
wobj = pandas.DataFrame(
    numpy.array([ind.fitness.wvalues for ind in pop]) * -1,
    columns=Y_train.columns)

ref_deap = numpy.max(wobj, axis=0) + 1

obj_scaled_leme = pandas.DataFrame(
    y_scaler.transform(obj), columns=Y_train.columns)

wobj_scaled_leme = obj_scaled_leme * y_weights

obj_scaler = MinMaxScaler()

obj_scaled_self = pandas.DataFrame(
    obj_scaler.fit_transform(obj), columns=Y_train.columns)

Y_train_scaled_deap = pandas.DataFrame(
    obj_scaler.transform(Y_train), columns=Y_train.columns)

Y_train_scaled_deap_weighted = Y_train_scaled_deap * y_weights

wobj_scaled_self = obj_scaled_self * y_weights

ref_deap_scaled = numpy.max(wobj_scaled_self, axis=0) + 1


def main():
    print("deap:", timestamp, model, seed_deap, gen, "leme:", seed_leme)

    # TODO: reevaluate the relevance of raw hypervolume.
    # print("raw hv:")
    # print("leme:", hv.hypervolume(Y_train_weighted.to_numpy(), ref_leme.to_numpy()))
    # print("deap:", hv.hypervolume(wobj.to_numpy(), ref_deap.to_numpy()))

    print("\nleme-scaled hv:")
    print("leme:", hv.hypervolume(
        Y_train_scaled_self_weighted.to_numpy(),
        ref_leme_scaled.to_numpy()))
    print("deap:", hv.hypervolume(
        wobj_scaled_leme.to_numpy(),
        ref_leme_scaled.to_numpy()))

    # TODO: reevaluate the relevance of scaling the
    # hypervolume to the deap result.
    # print("\ndeap-scaled hv:")
    # print("leme:", hv.hypervolume(
    #     Y_train_scaled_deap_weighted.to_numpy(),
    #     ref_deap_scaled.to_numpy()))
    # print("deap:", hv.hypervolume(
    #     wobj_scaled_self.to_numpy(),
    #     ref_deap_scaled.to_numpy()))


if __name__ == "__main__":
    main()
