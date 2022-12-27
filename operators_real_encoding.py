import numpy as np
from numpy.random import default_rng

rng = default_rng()


def mutation_gaussian(chromosome):
    sigma = np.divide(np.abs(np.mean(chromosome)), len(chromosome))
    return chromosome + rng.normal(0, sigma, len(chromosome))


def mutation_cauchy(chromosome):
    return chromosome + np.random.standard_cauchy(len(chromosome))


def xover_single_point(chromosome1, chromosome2):
    num_of_genes = len(chromosome1)

    # numpy.Generator.integers(low, high) low inclusive, high exclusive
    cross_idx = rng.integers(1, num_of_genes) if num_of_genes > 1 else 1

    child1 = np.hstack((chromosome1[0: cross_idx], chromosome2[cross_idx:]))
    # child2 = np.hstack((chromosome2[0: cross_idx], chromosome1[cross_idx:]))

    return child1  # , child2


def xover_two_point(chromosome1, chromosome2):
    ...


def xover_average(chromosome1, chromosome2):
    assert len(chromosome1) == len(chromosome2)

    return (chromosome1 + chromosome2) / 2


def xover_arithmetic(chromosome1, chromosome2):
    assert len(chromosome1) == len(chromosome2)

    r = rng.uniform(0, 1)
    return r * chromosome2 + (1 - r) * chromosome2


def xover_flat(chromosome1, chromosome2):
    ...


def xover_blend(chromosome1, chromosome2):
    ...


def xover_simplex(chromosome1, chromosome2):
    ...


def xover_unimodal_normal_distribution(chromosome1, chromosome2):
    ...


def xover_parent_centric(chromosome1, chromosome2):
    ...
