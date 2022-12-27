from objective_functions import *
from operators_real_encoding import *
from function_tools import *

import numpy as np

from numpy.random import default_rng

rng = default_rng()

rng = default_rng()


def swap(I, i, j):
    tmp = I[j]
    I[j] = I[i]
    I[i] = tmp


def stochastic_ranking(fitness, scv, p_f=0.45):
    '''
    "Stochastic Ranking" procedure
    for sorting of individuals in the population based on the objective function and constraint violation simultaneously
    as described in:
    https://www.researchgate.net/publication/3418601_Yao_X_Stochastic_ranking_for_constrained_evolutionary_optimization_IEEE_Trans_Evol_Comput_4_284-294

    Input: population, their evaluation = [f(x), sum(Gi(x))_i=<1, m+p> ] = [fitness, sum of constrains violations]
    Output: sorted population from best to worst one

    :param scv: summed constrained violations
    :param p_f: Probability of comparing two individuals only by their fitness. For both feasible ones p_f = 1.


    Note:
    -----------------------
    Use it with rank-based tournament selection within EA with generational replacement strategy.
    ----------------------
    '''
    population_size = len(fitness)

    indices = np.arange(population_size)

    for i in range(population_size):
        no_swap = True
        for j in range(population_size - 1):
            u = rng.uniform(0, 1)
            if (scv[indices[j]] == 0 and scv[indices[j + 1]] == 0) or u < p_f:
                if fitness[indices[j]] > fitness[indices[j + 1]]:
                    swap(indices, j, j + 1)
                    no_swap = False
            else:
                if scv[indices[j]] > scv[indices[j + 1]]:
                    swap(indices, j, j + 1)
                    no_swap = False

        if no_swap:
            break
        else:
            no_swap = True

    return indices


def parent_selection(population_fitness, population_svc, params):
    '''
    Rank-based tournament selection using on "Stochastic Ranking" of the population.

    :returns parent_indices: to population of chosen parents
    '''
    parent_indices = np.zeros(params["parents_size"], dtype=np.int32)

    ranks = stochastic_ranking(population_fitness, population_svc)

    prob = 0.5
    probs = prob * (1 - prob) ** np.arange(0, params["tournament_size"])
    probs /= np.sum(probs)  # normalize to sum up to 1.

    for i in range(params["parents_size"]):
        in_tournament_indices = rng.integers(0, len(population_fitness), params["tournament_size"])
        sorted_by_rank = sorted(ranks[in_tournament_indices], key=lambda x: np.where(ranks == x)[0][0])
        parent_indices[i] = rng.choice(sorted_by_rank, p=probs)

    return parent_indices


def replacement_strategy(population, offspring, population_fitness, offspring_fitness,
                         population_scv, offspring_scv, mode="generational"):
    if mode == "generational":
        assert len(offspring_fitness) >= len(population_fitness)

        rank = stochastic_ranking(offspring_fitness, offspring_scv)
        ranked_offspring = offspring[rank]
        return ranked_offspring[0: len(population_fitness)]


def ea_stochastic_ranking(params, stats):
    generation_count = 0

    population = init_population(params)

    population_objectives = evaluate_specimen(population, params)
    population_fitness, population_scv = population_objectives[:, 0], population_objectives[:, 1]

    while generation_count < params["generations"]:
        if params["verbose"]:
            get_stats(population, generation_count, params, population_objectives, stats)

        parents_indices = parent_selection(population_fitness, population_scv, params)

        offspring = breed(population, parents_indices, params)

        offspring_objectives = evaluate_specimen(offspring, params)
        offspring_fitness, offspring_scv = offspring_objectives[:, 0], offspring_objectives[:, 1],

        population = replacement_strategy(population, offspring,
                                          population_fitness, offspring_fitness,
                                          population_scv, offspring_scv)

        population_objectives = evaluate_specimen(population, params)
        population_fitness, population_scv = population_objectives[:, 0], population_objectives[:, 1]

        generation_count += 1


if __name__ == "__main__":
    objective_function = ObjectiveFunction(g06_name)
    params = {'fitness_func': objective_function.fitness_func,
              'feasibility_func': objective_function.feasibility_func,
              # 'objectives_size': g11_objectives_size,
              'variables_size': objective_function.variables_size,
              'objectives_size': 2,
              'solution_bounds': [-10, 10],
              'population_size': 80,
              'offspring_size': 200,
              'parents_size': 80,
              'generations': 10,
              'initialize_solution_func': init_population,
              'replacement_strategy': "generational",
              'tournament_size': 10,
              'crossover': xover_arithmetic,
              'mutation': mutation_cauchy,
              'mutation_prob': 0.3,
              'verbose': True,
              }

    stats = Stats(params)

    ea_stochastic_ranking(params, stats)

    visualise_stats(stats, params, objective_function)

    debug = 1
