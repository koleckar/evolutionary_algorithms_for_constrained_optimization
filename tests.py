import numpy as np
from function_tools import *
from ea_stochastic_ranking import *
from ea_nsga2 import *
from ea_spea2 import *


def test_spea2_density():
    ...


def test_spea2_raw_fitness():
    f = np.array([[1, 1],
                  [2, 3],
                  [1, 3],
                  [3, 2],
                  [4, 4]])

    raw_fitness, strengths = get_raw_fitness_and_strengths(f)

    strengths_gold = np.array([4, 1, 2, 1, 0])
    raw_fitness_gold = np.array([0, 6, 4, 4, 8])

    assert np.array_equal(strengths_gold, strengths), "strength err"
    assert np.array_equal(raw_fitness_gold, raw_fitness), "raw_fitness err"


def test_nsga2_crowding_distance():
    params = {"population_size": 6,
              "objectives_size": 2}
    population_objectives = np.array([[6, 5], [3, 5], [8, 9], [3, 3], [5, 4], [7, 4]])
    cro = get_crowding_distances(population_objectives)


def test_nsga2_non_dominated_front():
    params = {"population_size": 6,
              "objectives_size": 2}
    population_objectives = np.array([[6, 5], [3, 5], [8, 9], [3, 3], [5, 4], [7, 4]])
    fronts = get_non_dominated_fronts(population_objectives)
    assert fronts == np.array([3, 2, 4, 1, 2, 3])


def test_nsga2_custom_sort_for_tournament():
    idcs = np.array([2, 2, 0, 1, 0, 3])
    fronts = np.array([3, 1, 2, 1])
    crowdings = np.array([0.1, 0.102, 32, np.inf])

    print("idcs:", idcs)
    # print("fronts:", fronts[idcs])
    my_sort(idcs, fronts, crowdings, binary_tournament_cmp)
    print("sorted:", idcs)

    debug = 1


def test_nsga2_crowding_fronts_2():
    f = np.array([[0.8836321167084642, 0.5607133376216897, 0.02381710871203957],
                  [0.5108776119588554, 0.737363832069241, 0.8355729415608875],
                  [0.0804380750956224, 0.790857997792852, 0.28702306060277794],
                  [0.907569194322823, 0.8328436077371422, 0.9411191834152739],
                  [0.5482773927964455, 0.6756974050554325, 0.4701658371135],
                  [0.9725002352331573, 0.46457376137259554, 0.26412896062456337],
                  [0.5185816618886874, 0.09791241553737562, 0.09226819824384669]])
    fronts_gold = np.array([1, 1, 1, 3, 2, 2, 1])
    crowdings_gold = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    fronts = get_non_dominated_fronts(f)
    crowdings = np.round(get_crowding_distances(f), 2)

    print(fronts_gold)
    print(fronts)

    print(crowdings_gold)
    print(crowdings)

    assert np.array_equal(fronts_gold, fronts), "fronts fail."
    assert np.array_equal(crowdings_gold, crowdings), "crowdings fail."


def test_nsga2_crowding_fronts_1():
    f = np.array([[0.025102722835371738, 0.13211493983286182],
                  [0.7114048038690621, 0.4984216541102089],
                  [0.26930105112871994, 0.4823660085390846],
                  [0.2004787209032317, 0.9742757849985806],
                  [0.525909791663457, 0.9170813020388819],
                  [0.4811523772981978, 0.526614730637968],
                  [0.21400461912640778, 0.556870788900304]])
    fronts_gold = np.array([1, 3, 2, 2, 4, 3, 2])
    crowdings_gold = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.6843842446699838])

    crowdings = np.round(get_crowding_distances(f), 2)
    fronts = get_non_dominated_fronts(f)

    print(fronts_gold)
    print(fronts)

    print(crowdings_gold)
    print(crowdings)

    assert np.array_equal(fronts_gold, fronts)
    assert np.array_equal(crowdings_gold, crowdings), "crowdings fail."

    debug = 1


if __name__ == "__main__":
    ...
