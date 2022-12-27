import sys
from typing import overload

import numpy as np
from matplotlib import pyplot as plt

from numpy.random import default_rng

rng = default_rng()


def init_population(params):
    low, high = params["solution_bounds"]
    return rng.uniform(low, high, [params["population_size"], params["variables_size"]])


def x_dominates_y(x, y, objectives):
    return np.all(objectives[x, :] <= objectives[y, :]) and np.any(objectives[x, :] < objectives[y, :])


def x_dominates_y_2(x, objectives_x, y, objectives_y):
    return np.all(objectives_x[x, :] <= objectives_y[y, :]) and np.any(objectives_x[x, :] < objectives_y[y, :])


def get_distinct_dict(population):
    distinct_dict = {}
    for i, x in enumerate(population):
        if distinct_dict.__contains__(x.tobytes()):
            continue
        else:
            distinct_dict[x.tobytes()] = 1
        for j, y in enumerate(population):
            if i == j:
                continue
            if np.array_equal(x, y):
                distinct_dict[x.tobytes()] += 1
    return distinct_dict


def evaluate_specimen(specimen, params, mode="single_objective"):
    size = specimen.shape[0]
    objectives = np.zeros([size, params["objectives_size"]])

    if mode == "single_objective":
        for i, x in enumerate(specimen):
            vals = params["fitness_func"](*x)
            f = vals[0]
            cv = vals[1:]
            objectives[i, 0] = f
            objectives[i, 1] = np.sum(cv)
            assert objectives[i, 1] >= 0, "scv is < 0 !"

    elif mode == "multi_objective":
        for i, x in enumerate(specimen):
            objectives[i, :] = params["fitness_func"](*x)

    else:
        assert 1 == 0, 'wrong mode'

    return objectives


def breed(population, parents_indices, params):
    offspring = np.zeros([params["offspring_size"], params["variables_size"]])

    for i in range(params["offspring_size"]):
        j, k = rng.integers(0, len(parents_indices), 2)
        parent1 = population[parents_indices[j]]
        parent2 = population[parents_indices[k]]

        child = params["crossover"](parent1, parent2)

        if rng.uniform(0, 1) > params["mutation_prob"]:
            mutant = params["mutation"](child)
            offspring[i] = mutant
        else:
            offspring[i] = child

    # print("distinct_offspring: ", len(get_distinct_dict(offspring)), "/", params["offspring_size"])

    return offspring


class Stats:
    def __init__(self, params):
        self.params = params
        self.generations = []
        self.feasible = []
        self.distinct = []
        self.fitness = []
        self.scv = []


def print_stats(generation_count, population, population_fitness, population_scv, best_idx,
                feasible, infeasible, distinct_dict):
    print(generation_count, ":  ",
          np.round(population[best_idx], 2), "  ",
          "f  =", np.round(population_fitness[best_idx], 3), "  ",
          "scv=", np.round(population_scv[best_idx], 2), "  ",
          "feasible:", str(feasible) + "/" + str(feasible + infeasible), "  ",
          "unique:", len(distinct_dict), "/", population.shape[0],
          file=sys.stderr)


# todo: if multiple solutions have scv==0, how to choose the best one?
def get_stats(population, generation, params, objectives, stats=None):
    feasible, infeasible = 0, 0
    feasible_idcs = []
    for i, x in enumerate(population):
        if params["feasibility_func"](*x):
            feasible += 1
            feasible_idcs.append(i)
        else:
            infeasible += 1

    distinct_dict = get_distinct_dict(population)

    population_fitness = objectives[:, 0]
    population_scv = objectives[:, 1] if params["objectives_size"] == 2 else np.sum(objectives[:, 1:], axis=1)

    if feasible > 0:
        best_idx = feasible_idcs[0]  # take the first one
    else:
        best_idx = np.argmin(population_scv)

    # update Stats
    if stats is not None:
        stats.generations.append(generation)
        stats.feasible.append(feasible)
        stats.distinct.append(len(distinct_dict))
        stats.fitness.append(population_fitness[best_idx])
        stats.scv.append(population_scv[best_idx])

    print_stats(generation, population, population_fitness, population_scv, best_idx,
                feasible, infeasible, distinct_dict)


# todo: legends, optimum gap
def visualise_stats(stats_over_runs, params, objective_function, algorithm_name, number_of_runs):
    symbols = ["x", "o", ">", "<", "s", "p", "*", "h", "H", "+", "x", "o", ">", "<", "s", "p", "*", "h", "H", "+"]
    # colors = ["c", "m", "y", "k", "b"]
    colors = ["y"] * 20

    # ------ Fitness plot -------
    max_fitness_call = 0
    for j, stats in enumerate(stats_over_runs):
        fitness_calls = np.array(stats_over_runs[j].generations) * (
                params["population_size"] + params["offspring_size"])
        max_fitness_call = fitness_calls[-1] if fitness_calls[-1] > max_fitness_call else max_fitness_call
        for i, scv in enumerate(stats.scv):
            if np.abs(scv) - 0.000015 < 0:
                plt.plot(fitness_calls[i], stats.fitness[i], 'b' + symbols[j], markersize=1)
            else:
                plt.plot(fitness_calls[i], stats.fitness[i], 'r' + symbols[j], markersize=1)
        plt.plot(fitness_calls, np.array(stats.fitness), colors[j] + '-', linewidth=0.4)
    plt.hlines(objective_function.opt, 0, max_fitness_call,
               colors="g", linestyles='dashed', linewidth=0.7, label="opt=" + str(objective_function.opt))

    plt.xlabel("fitness evaluations = generations * (offspring_size + population_size)")
    plt.ylabel("fitness of the best specimen")
    plt.title(objective_function.name + "  -  " + algorithm_name + "  -  optimum gap(average): "
              + str(get_average_optimality_gap(stats_over_runs, objective_function)) + "%")
    plt.ylim(objective_function.opt + 1 / 4 * objective_function.opt,
             objective_function.opt - 1 / 4 * objective_function.opt)
    plt.plot(-np.inf, np.inf, 'r' + symbols[j], markersize=3, label="is infeasible")
    plt.plot(-np.inf, np.inf, 'b' + symbols[j], markersize=3, label="is feasible")
    plt.legend()

    plt.savefig("plots/" + objective_function.name + "_fitness_"
                + algorithm_name + "_runs_" + str(number_of_runs))
    plt.show()

    # ------ feasible / unique -------
    for j, stats in enumerate(stats_over_runs):
        fitness_calls = np.array(stats_over_runs[j].generations) * (
                params["population_size"] + params["offspring_size"])
        plt.plot(fitness_calls, stats.feasible, 'g', linewidth=0.5)
        plt.plot(fitness_calls, stats.distinct, 'b', linewidth=0.2)

    plt.plot(-np.inf, np.inf, 'g-', label="feasible")
    plt.plot(-np.inf, np.inf, 'b-', label="unique")
    plt.ylim(0, params["population_size"])
    plt.xlabel("fitness evaluations")
    plt.ylabel("fesible / unique")
    plt.title(objective_function.name + "  -  " + algorithm_name)
    plt.legend()
    plt.savefig("plots/" + objective_function.name + "_feasible_unique_"
                + algorithm_name + "_runs_" + str(number_of_runs))
    plt.show()

    # plt.plot(fitness_calls, stats.scv, 'r')
    # plt.xlabel("fitness evaluations")
    # plt.ylabel("scv")
    # plt.show()


def get_average_optimality_gap(stats_over_runs, objective_function):
    fit_sum = 0
    for stats in stats_over_runs:
        fit_sum += stats.fitness[-1]
    fit_sum /= len(stats_over_runs)
    return np.round(np.abs((objective_function.opt - fit_sum) / objective_function.opt) * 100, 2)


def visualise_domain(objective_function, solution):
    plt.rcParams.update({
        "axes.facecolor": (1.0, 0.0, 0.0, 0.3),  # green with alpha = 50%
    })
    poly = [[1, 1], [2, 1], [2, 2], [1, 2], [0.5, 1.5]]
    b = poly + [poly[0]]
    a = np.column_stack(poly + [poly[0]])
    plt.plot(*np.column_stack(poly + [poly[0]]))
    #
    # x1, x2 = np.meshgrid(np.linspace(-100, 100, 200), np.linspace(-100, 100, 200))
    # f = (x1 - 10) ** 3 + (x2 - 20) ** 3
    # g1 = -(x1 - 5) ** 2 - (x2 - 5) ** 2 + 100
    # plt.plot(14.095, 0.843, 'gx')
    # plt.contour(x1, x2, f)
    # plt.contour(x1, x2, g1)
    plt.grid()
    plt.show()

    d = np.linspace(-2, 16, 300)
    x, y = np.meshgrid(d, d)
    plt.imshow(((y >= 2) & (2 * y <= 25 - x) & (4 * y >= 2 * x - 8) & (y <= 2 * x - 5)).astype(int),
               extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Blues", alpha=0.5)

    # plot the lines defining the constraints
    x = np.linspace(-100, 100, 200)
    # y >= 2
    y1 = (x * 0) + 2
    # 2y <= 25 - x
    y2 = (25 - x) / 2.0
    # 4y >= 2x - 8
    y3 = (2 * x - 8) / 4.0
    # y <= 2x - 5
    y4 = 2 * x - 5

    # Make plot
    plt.plot(x, 2 * np.ones_like(y1))
    plt.plot(x, y2, label=r'$2y\leq25-x$')
    plt.plot(x, y3, label=r'$4y\geq 2x - 8$')
    plt.plot(x, y4, label=r'$y\leq 2x-5$')
    plt.xlim(0, 16)
    plt.ylim(0, 11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()

    # g1 = -(x1 - 5) ** 2 - (x2 - 5) ** 2 + 100
    # g2 = (x1 - 6) ** 2 + (x2 - 5) ** 2 - 82.81
    #
    # g3 = np.maximum(0, 13 - x1)
    # g4 = np.maximum(0, x1 - 100)
    # g5 = np.maximum(0, 0 - x2)
    # g6 = np.maximum(0, x2 - 100)


if __name__ == "__main__":
    visualise_domain(None, None)
