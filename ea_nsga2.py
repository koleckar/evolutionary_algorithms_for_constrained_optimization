from objective_functions import *
from operators_real_encoding import *
from function_tools import *

import numpy as np


def get_one_front(P, objectives):
    F = {P.pop()}
    non_sorted_indices = list(P)
    was_dominated = set()

    for x in non_sorted_indices:
        if x in F or x in was_dominated:
            continue

        not_dominated = True
        move_from_F_to_P = set()

        for y in F:

            if x_dominates_y(x, y, objectives):
                was_dominated.add(y)
                move_from_F_to_P.add(y)
            elif x_dominates_y(y, x, objectives):
                not_dominated = False
                was_dominated.add(x)
                break
            else:
                pass
                # non-dominate each other

        for y in move_from_F_to_P:
            F.remove(y)
            P.add(y)

        if not_dominated:
            P.remove(x)
            F.add(x)

    return F, P


def get_non_dominated_fronts(objectives):
    '''
    https://sci-hub.se/10.1162/evco.2008.16.3.355
    '''

    fronts = np.zeros(objectives.shape[0], dtype=np.int32)
    front = 0
    P = set(np.arange(0, objectives.shape[0]))
    while len(P) > 0:
        front += 1
        F, P = get_one_front(P, objectives)
        for f in F:
            fronts[f] = front

    return fronts


def get_front_i_members(fronts, i):
    front_i = set()
    for j in range(len(fronts)):
        if fronts[j] == i:
            front_i.add(j)
    return front_i


def replacement_strategy(population, offspring, population_objectives, offspring_objectives, params):
    combined_specimen = np.vstack([population, offspring])
    combined_objectives = np.vstack([population_objectives, offspring_objectives])

    # print("distinct: ", len(get_distinct_dict(combined_specimen)), "/", combined_specimen.shape[0])

    fronts = get_non_dominated_fronts(combined_objectives)

    new_population = set()
    i = 1
    front_i = get_front_i_members(fronts, i)
    while len(new_population) + len(front_i) < params["population_size"]:
        new_population = new_population.union(front_i)
        i += 1
        front_i = get_front_i_members(fronts, i)

    front_i_array = np.array(list(front_i))
    crowding_front_i = get_crowding_distances(combined_specimen[front_i_array, :],
                                              combined_objectives[front_i_array, :])
    free_space = params["population_size"] - len(new_population)
    indices = sorted(np.arange(0, len(crowding_front_i)),
                     key=lambda x: crowding_front_i[x], reverse=True)[0: free_space]

    new_population = new_population.union(set(front_i_array[indices]))

    return combined_specimen[list(new_population), :], combined_objectives[list(new_population), :]


def get_crowding_distances(population, objectives):
    '''
    For each objective sort indices by this objective, compute distances, average.
    '''
    specimen_size, objectives_size = objectives.shape
    partial_crowd_dists = np.zeros(objectives.shape)
    crowd_dists = np.zeros(specimen_size)

    for i in range(objectives_size):
        sorted_indices = sorted(np.arange(0, specimen_size), key=lambda y: objectives[y, i], reverse=True)
        obj_i_max = objectives[sorted_indices[0], i]
        obj_i_min = objectives[sorted_indices[-1], i]

        if obj_i_min == obj_i_max:
            continue

        # todo: without for loops?
        for j in range(1, specimen_size - 1):
            x = (objectives[sorted_indices[j + 1], i] - objectives[sorted_indices[j - 1], i]) / (obj_i_max - obj_i_min)
            # x = np.sqrt((objectives[sorted_indices[j + 1], i] - objectives[sorted_indices[j - 1], i]) ** 2)
            # x = np.abs((objectives[sorted_indices[j + 1], i] - objectives[sorted_indices[j - 1], i])) / 2
            partial_crowd_dists[sorted_indices[j], i] = x

        c = objectives[sorted_indices, i]
        partial_crowd_dists[sorted_indices[0], i] = np.inf
        partial_crowd_dists[sorted_indices[-1], i] = np.inf

    for i in range(specimen_size):
        crowd_dists[i] = np.sum(partial_crowd_dists[i, :])

    # Penalize crowding distance of identical specimen?
    # for i, x in enumerate(population):
    #     for j, y in enumerate(population):
    #         if i == j:
    #             continue
    #         if np.array_equal(x, y):
    #             crowd_dists[i] = 0
    #             crowd_dists[j] = 0
    #             break

    return crowd_dists


def binary_tournaments(front_x, front_y, crowding_x, crowding_y):
    '''
    :return: True, if solution x is better than y
    '''
    if front_x < front_y:
        return True
    elif front_x == front_y and crowding_x > crowding_y:
        return True
    else:
        return False


def binary_tournaments_constraint_domination(is_feasible_x, is_feasible_y, scv_x, scv_y, x_dominates_y):
    if is_feasible_x and not is_feasible_y:
        return True

    if is_feasible_x and is_feasible_y:
        return x_dominates_y

    if not is_feasible_x and not is_feasible_y:
        return scv_x < scv_y

    return False


def binary_tournament_cmp(i, j, front, crowding):
    if front[i] < front[j]:
        return True
    elif front[i] == front[j] and crowding[i] > crowding[j]:
        return True
    else:
        return False


def swap(i, j, idcs):
    tmp = idcs[i]
    idcs[i] = idcs[j]
    idcs[j] = tmp


def my_sort(indices, front, crowding, cmp):
    for i in range(len(indices) - 1):
        for j in range(len(indices) - i - 1):
            if not cmp(indices[j], indices[j + 1], front, crowding):
                swap(j, j + 1, indices)


def my_sort_cd(indices, front, crowding, cvs, objectives, cmp):
    for i in range(len(indices) - 1):
        for j in range(len(indices) - i - 1):
            if not cmp(cvs[i] == 0, cvs[j] == 0, cvs[i], cvs[j], x_dominates_y(i, j, objectives)):
                swap(j, j + 1, indices)


def parent_selection(crowdings, fronts, objectives, params):
    '''
    Rank-based tournament selection using on "non-dominated fronts" of the population as fitness.

    :returns parent_indices: to population of chosen parents
    '''

    cvs = np.sum(objectives[:, 1:], axis=1)

    parent_indices = np.zeros(params["parents_size"], dtype=np.int32)
    prob = 0.5
    probs = prob * (1 - prob) ** np.arange(0, params["tournament_size"])
    probs /= np.sum(probs)  # normalize to sum up to 1.

    for i in range(params["parents_size"]):
        in_tournament_indices = rng.integers(0, params["population_size"], params["tournament_size"])

        if params['cmp_mode'] == "constraint_domination":
            my_sort_cd(in_tournament_indices, fronts, crowdings, cvs, objectives,
                       binary_tournaments_constraint_domination)
        elif params['cmp_mode'] == "binary_tournament":
            my_sort(in_tournament_indices, fronts, crowdings, binary_tournament_cmp)
        else:
            RuntimeError("use: params['cmp_mode'] = constraint_domination / binary_tournament")

        parent_indices[i] = rng.choice(in_tournament_indices, p=probs)

    return parent_indices


def ea_nsga2(params, stats):
    generation_count = 0

    population = init_population(params)
    population_objectives = evaluate_specimen(population, params, params["evaluation_mode"])

    while generation_count < params["generations"]:
        if params["verbose"]:
            get_stats(population, generation_count, params, population_objectives, stats)

        population_crowding = get_crowding_distances(population, population_objectives)
        population_fronts = get_non_dominated_fronts(population_objectives)
        parents_indices = parent_selection(population_crowding, population_fronts, population_objectives, params)

        offspring = breed(population, parents_indices, params)
        offspring_objectives = evaluate_specimen(offspring, params, params["evaluation_mode"])

        population, population_objectives \
            = replacement_strategy(population, offspring, population_objectives, offspring_objectives, params)

        if generation_count > 100:
            if stats.fitness[generation_count] == stats.fitness[generation_count - 100]:
                break

        generation_count += 1


def main():
    params_nsga2 = {'fitness_func': g06,
                    'feasibility_func': g06_feasibility,
                    'variables_size': g06_variables_size,
                    # 'objectives_size': g05_objectives_size,
                    # 'evaluation_mode': "multi_objective",
                    "objectives_size": 2,
                    'evaluation_mode': "single_objective",
                    'solution_bounds': [-1000, 1000],
                    'population_size': 100,
                    'offspring_size': 400,
                    'parents_size': 140,
                    'generations': 300,
                    'initialize_solution_func': init_population,
                    'cmp_mode': "constraint_domination",  # constraint_domination  / "binary_tournament"
                    'tournament_size': 30,
                    'crossover': xover_arithmetic,
                    'mutation': mutation_cauchy,
                    'mutation_prob': 0.4,
                    'replacement_strategy': "generational",
                    'diversity_ratio': 0.4,
                    'verbose': True,
                    }

    print(params_nsga2)
    ea_nsga2(params_nsga2)


if __name__ == "__main__":
    main()

# =======================
# def non_dominated_fronts(objectives, params):
#     fronts = np.zeros(params["population_size"])
#     idle = set(np.arange(0, params["population_size"]))
#     front = 0
#     while True:
#         front += 1
#         for i in range(params["population_size"]):
#             if fronts[i] == 0:
#                 not_dominated = True
#                 for j in range(params["population_size"]):
#                     if not_dominated:
#                         if fronts[j] == 0 or fronts[j] == front:
#                             for k in range(params["objectives_size"]):
#                                 if objectives[i, k] < objectives[j, k]:
#                                     not_dominated = False
#                     else:
#                         break
#                 if not_dominated:
#                     fronts[i] = front
#                     idle.remove(i)
#
#         if idle.__sizeof__() == 0:
#             break
#
#     return fronts


# todo: solve the extreme values -> crowding = inf
# def crowding_distance(objectives, params):
#     '''
#     objectives = np.ndarray [population_size, objectives_size]
#     '''
#     crowding_number = np.zeros(params["population_size"])
#
#     predecessors_indices = np.ones(objectives.shape) * -1
#     successors_indices = np.ones(objectives.shape) * -1
#
#     for i in range(params["objectives_size"]):
#
#         for j, objective_i_of_specimen_j in enumerate(objectives[:, i]):
#
#             for k, objective_i_of_specimen_k in enumerate(objectives[:, i]):
#
#                 if j == k:
#                     continue
#
#                 # todo: predec/succes indices are set to -1 !
#                 if objective_i_of_specimen_j > objective_i_of_specimen_k > objectives[predecessors_indices[j, i], i]:
#                     predecessors_indices[j, i] = k
#
#                 if objective_i_of_specimen_j < objective_i_of_specimen_k < objectives[successors_indices[j, i], i]:
#                     successors_indices[j, i] = k
#
#             # todo: where to assign crowding distance = inf
#             #    can do if any predecessor/successor idx == -1 ->crowding=inf
#             if predecessors_indices[i, j] == -1:
#                 crowding_number[j] = np.inf
#
#     for specimen_idx in range(params["population_size"]):
#         if crowding_number[specimen_idx] == np.inf:
#             continue
#
#         for i in range(params["objectives_size"]):
#             predecessor_objective_i = objectives[predecessors_indices[specimen_idx, i], i]
#             successor_objective_i = objectives[successors_indices[specimen_idx, i], i]
#
#             crowding_number[specimen_idx] += successor_objective_i - predecessor_objective_i
#
#     return crowding_number
