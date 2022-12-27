from objective_functions import *
from operators_real_encoding import *
from function_tools import *
from ea_stochastic_ranking import ea_stochastic_ranking
from ea_nsga2 import ea_nsga2
from ea_spea2 import ea_spea2
import pickle


def multiple_runs():
    stats_over_runs = []

    for i in range(number_of_runs):
        stats = Stats(params)
        if algorithm_name == "sr":
            ea_stochastic_ranking(params, stats)
        elif algorithm_name == "nsga2":
            ea_nsga2(params, stats)
        elif algorithm_name == "spea2":
            ea_spea2(params, stats)
        else:
            RuntimeError("Wrong ea algorithm chosen. Use: sr / nsga2 .")
        stats_over_runs.append(stats)

    return stats_over_runs


if __name__ == "__main__":
    load_stats = False

    number_of_runs = 5

    # for algorithm_name in ["nsga2", "sr"]:
    for algorithm_name in ["nsga2"]:
        # for objective_function_name in [g05_name, g06_name, g08_name, g11_name, g24_name]:
        for objective_function_name in [g05_name]:
            objective_function = ObjectiveFunction(objective_function_name)
            params = {'fitness_func': objective_function.fitness_func,
                      'feasibility_func': objective_function.feasibility_func,
                      'objectives_size': 2,  # objective_function.objectives_size
                      'variables_size': objective_function.variables_size,
                      'evaluation_mode': "single_objective",
                      'solution_bounds': [-1000, 1000],
                      'population_size': 50,
                      'offspring_size': 150,
                      'parents_size': 40,
                      'generations': 2000,
                      'initialize_solution_func': init_population,
                      'cmp_mode': "binary_tournament",  # constraint_domination  / "binary_tournament"
                      'replacement_strategy': "generational",
                      'tournament_size': 15,
                      'crossover': xover_arithmetic,
                      'mutation': mutation_cauchy,
                      'mutation_prob': 0.35,
                      'verbose': True,
                      }

            if load_stats:
                # todo: some of older stats.pickle files are only stats_over_runs, not the dict, rerun?
                with open("stats/" + objective_function.name + "_stats_over_" + str(number_of_runs) + "_runs_"
                          + algorithm_name + ".pickle", "rb") as file:
                    stats = pickle.load(file)
                    stats_over_runs = stats["stats_over_runs:"]
                    params = stats["params"]
                    objective_function = stats["objective_function"]
            else:
                stats_over_runs = multiple_runs()

                with open("stats/" + objective_function.name + "_stats_over_" + str(number_of_runs) + "_runs_"
                          + algorithm_name + ".pickle", "wb") as f:
                    pickle.dump({"stats_over_runs:": stats_over_runs,
                                 "params": params,
                                 "objective_function": objective_function},
                                f)

            visualise_stats(stats_over_runs, params, objective_function, algorithm_name, number_of_runs)
