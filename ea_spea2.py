import numpy as np
from function_tools import *
from objective_functions import *
from operators_real_encoding import *


def update_k_closest_dists(dists, idx, dist, K):
    """
    Find if the dist is among the k closest dists. if yes find 'j' such: dists[idx, i] < dist < dist[idx, j]
    put dists[idx, j] = dist , and move the rest to one the right
    """
    found = False
    for k in range(K):
        if dist < dists[idx, k]:
            found = True
            break

    if found:
        while k < K:
            tmp = dists[idx, k]
            dists[idx, k] = dist
            dist = tmp
            k += 1


def find_kth_closest_dists(fitness, invalid_indices, K):
    """
    :param fitness: np.ndarray [archive + non-dominated population, objective_size]
    :param invalid_indices: bool np.ndarray of specimen to be removed from new archive
    :param K: if the ties have to be broken, in next iteration judge by k-th closest dist in f-space
    :return: None, update 1 index in 'invalid_indices'
    """
    kth_closest_dists = np.ones([fitness.shape[0], K]) * np.inf

    for i, f_i in enumerate(fitness):
        if invalid_indices[i]:
            continue

        for j, f_j in enumerate(fitness):
            if invalid_indices[j] or i == j:
                continue

            dist = np.linalg.norm(f_i - f_j)
            update_k_closest_dists(kth_closest_dists, i, dist, K)

    return np.where(kth_closest_dists[:, K - 1] == np.argmin(kth_closest_dists[:, K - 1]))[0]


def find_min_dist_idx(coordinates, invalid_indices):
    k = 1

    min_solution = find_kth_closest_dists(coordinates, invalid_indices, k)

    if len(min_solution) > 1:
        min_solutions = min_solution

        mask = np.ones(len(invalid_indices), dtype=bool)  # The min solutions are valid, so can work with mask so forth.
        mask[min_solutions] = False

        while len(min_solutions) != 1:
            k += 1

            min_solutions_next = find_kth_closest_dists(coordinates, mask, k)

            for idx_i in min_solutions:
                found = False
                for idx_j in min_solutions_next:
                    if idx_i == idx_j:
                        found = True
                    if not found:
                        mask[idx_i] = True

            min_solutions = min_solutions_next

        min_solution = min_solutions

    invalid_indices[min_solution] = True


def get_idx_with_closest_dist(coordinates, invalid_indices):
    n = coordinates.shape[0]
    distances = np.ones([n, n]) * np.inf

    for i in range(n):
        if invalid_indices[i]:
            continue
        for j in range(i, n):
            if invalid_indices[j] or i == j:
                continue

            dist = np.linalg.norm(coordinates[i, :] - coordinates[j, :])
            distances[i, j] = dist
            distances[j, i] = dist

    min_dists = np.partition(distances, 0, axis=1)[:, 0]
    idx = np.argmin(min_dists)
    return idx


def environmental_selection(population, population_spea2_fitness, archive, archive_spea2_fitness, params):
    non_dominated_population_indices = np.where(population_spea2_fitness < 1)[0]
    non_dominated_archive_indices = np.where(archive_spea2_fitness < 1)[0]

    new_size = len(non_dominated_population_indices) + len(non_dominated_archive_indices)

    if new_size > params["archive_size"]:

        k = new_size - params["archive_size"]

        new_archive = np.vstack([archive[non_dominated_archive_indices, :],
                                 population[non_dominated_population_indices, :]])
        new_archive_fitness = np.hstack([archive_spea2_fitness[non_dominated_archive_indices],
                                         population_spea2_fitness[non_dominated_population_indices]])

        to_remove = np.zeros(new_archive.shape[0], dtype=bool)
        for i in range(k):
            idx = get_idx_with_closest_dist(new_archive, to_remove)
            to_remove[idx] = True

        return new_archive[~to_remove, :], new_archive_fitness[~to_remove]

    elif new_size < params["archive_size"]:
        k = params["archive_size"] - new_size

        mask = np.ones(population.shape[0], dtype=bool)
        mask[non_dominated_population_indices] = False

        for _ in range(k):
            idx_min = np.nan
            f_min = np.inf
            for i, can_choose_i in enumerate(mask):
                if can_choose_i:
                    if population_spea2_fitness[i] < f_min:
                        f_min = population_spea2_fitness[i]
                        idx_min = i

            mask[idx_min] = False

        return np.vstack([archive[non_dominated_archive_indices, :], population[~mask, :]]), \
               np.hstack([archive_spea2_fitness[non_dominated_archive_indices], population_spea2_fitness[~mask]])

    else:
        return np.vstack([archive[non_dominated_archive_indices, :],
                          population[non_dominated_population_indices, :]]), \
               np.hstack([
                   archive_spea2_fitness[non_dominated_archive_indices, :],
                   population_spea2_fitness[non_dominated_population_indices]])


# density: calculate distances
# sort by first, continue sorting by k-th if ties must be broken.
# todo: is density calculated in fitness or variable space?
def get_density(coordinates, k):
    n = coordinates.shape[0]

    dists = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue
            dist_i_j = np.linalg.norm(coordinates[i, :] - coordinates[j, :])
            dists[i, j] = dist_i_j
            dists[j, i] = dist_i_j

    # downside of np.partition is: it creates a new array.
    kth_closest_dists = np.partition(dists, k, axis=1)[:, k]  # index k is truly ok, [0] is always 0 (i, i)

    return 1 / (kth_closest_dists ** k + 2)


def get_raw_fitness_and_strengths(fitness):
    n = fitness.shape[0]

    dominates = np.zeros([n, n], dtype=bool)
    is_dominated = np.zeros([n, n], dtype=bool)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue
            if x_dominates_y(i, j, fitness):
                dominates[i, j] = True
                is_dominated[j, i] = True
            elif x_dominates_y(j, i, fitness):
                dominates[j, i] = True
                is_dominated[i, j] = True
            else:
                ...  # from the same non-dominated front

    strengths = np.sum(dominates, axis=1)

    raw_fitness = np.zeros(n, dtype=np.int)
    for i in range(n):
        raw_fitness[i] = np.sum(strengths[is_dominated[i, :]])

    return raw_fitness, strengths


def get_spea2_fitness(fitness, params):
    """
    :param fitness: Fitness of both population and archive.
    :return:
    """

    densities = get_density(fitness, params["k_for_density"])

    raw_fitness, _ = get_raw_fitness_and_strengths(fitness)

    return raw_fitness + densities


def parent_selection_binary_tournament(fitness, params):
    """
        Parent selection using binary tournament, smaller fitness wins.
        Fitness here is SPEA2 fitness = "raw fitness" + "density"

        :returns parent_indices: of chosen parents from archive
    """
    parent_indices = np.zeros(params["parents_size"], dtype=np.int)

    for i in range(params["parents_size"]):
        idx1, idx2 = rng.integers(0, fitness.shape[0], 2)
        parent_indices[i] = idx1 if fitness[idx1] < fitness[idx2] else idx2

    return parent_indices


def ea_spea2(params, stats):
    population = init_population(params)
    archive = np.ones([1, params["variables_size"]]) * np.inf

    generation_count = 0
    while generation_count < params["generations"]:

        population_objectives = evaluate_specimen(population, params, params["evaluation_mode"])
        archive_objectives = evaluate_specimen(archive, params, params["evaluation_mode"])

        spea2_fitness = get_spea2_fitness(np.vstack([population_objectives, archive_objectives]), params)
        population_spea2_fitness = spea2_fitness[0: population.shape[0]]
        archive_spea2_fitness = spea2_fitness[population.shape[0]:]

        archive, archive_spea2_fitness = environmental_selection(population, population_spea2_fitness,
                                                                 archive, archive_spea2_fitness,
                                                                 params)

        parent_indices = parent_selection_binary_tournament(archive_spea2_fitness, params)

        population = breed(archive, parent_indices, params)

        if params["verbose"]:
            archive_objectives = evaluate_specimen(archive, params, params["evaluation_mode"])
            get_stats(archive, generation_count, params, archive_objectives, stats)

        generation_count += 1


if __name__ == "__main__":
    # todo: population size == offspring size
    #   parents size <= archive size
    params_spea2 = {'k_for_density': int(np.sqrt(30 + 30)),  # todo: make by variables: population + archive
                    'archive_size': 30}

    objective_function = ObjectiveFunction(g06_name)
    params = {'fitness_func': objective_function.fitness_func,
              'feasibility_func': objective_function.feasibility_func,
              'objectives_size': 2,  # objective_function.objectives_size
              'variables_size': objective_function.variables_size,
              'evaluation_mode': "single_objective",
              'solution_bounds': [-1000, 1000],
              'population_size': 30,
              'offspring_size': 30,
              'parents_size': 25,
              'generations': 200,
              'initialize_solution_func': init_population,
              'replacement_strategy': "generational",
              'crossover': xover_arithmetic,
              'mutation': mutation_cauchy,
              'mutation_prob': 0.35,
              'verbose': True,
              }
    params.update(params_spea2)

    stats = Stats(params)
    ea_spea2(params, stats)
