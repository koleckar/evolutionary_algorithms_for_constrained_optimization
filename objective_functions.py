import numpy as np

'''
Static class representing the objective functions defined in
https://cw.fel.cvut.cz/wiki/_media/courses/a0m33eoa/cviceni/2006_problem_definitions_and_evaluation_criteria_for_the_cec_2006_special_session_on_constraint_real-parameter_optimization.pdf
Implemented functions: g06, g08, g11, g24
'''

eps = 0.00001


class ObjectiveFunction:
    def __init__(self, id):
        self.name = id

        if id == "g05":
            self.fitness_func = g05
            self.feasibility_func = g05_feasibility
            self.variables_size = g05_variables_size
            self.objectives_size = g05_objectives_size
            self.opt = g05_f_opt
        elif id == "g06":
            self.fitness_func = g06
            self.feasibility_func = g06_feasibility
            self.variables_size = g06_variables_size
            self.objectives_size = g06_objectives_size
            self.opt = g06_f_opt
        elif id == "g08":
            self.fitness_func = g08
            self.feasibility_func = g08_feasibility
            self.variables_size = g08_variables_size
            self.objectives_size = g08_objectives_size
            self.opt = g08_f_opt
        elif id == "g11":
            self.fitness_func = g11
            self.feasibility_func = g11_feasibility
            self.variables_size = g11_variables_size
            self.objectives_size = g11_objectives_size
            self.opt = g11_f_opt
        elif id == "g24":
            self.fitness_func = g24
            self.feasibility_func = g24_feasibility
            self.variables_size = g24_variables_size
            self.objectives_size = g24_objectives_size
            self.opt = g24_f_opt
        else:
            RuntimeError("Wrong objective function chosen. Use: g05, g06, g08, g11, g24.")


g05_objectives_size = 1 + 10 + 3
g06_objectives_size = 1 + 6
g08_objectives_size = 1 + 6
g11_objectives_size = 1 + 1 + 4
g24_objectives_size = 1 + 6

g05_variables_size = 4
g06_variables_size = 2
g08_variables_size = 2
g11_variables_size = 2
g24_variables_size = 2

g05_name = "g05"
g06_name = "g06"
g08_name = "g08"
g11_name = "g11"
g24_name = "g24"

g05_f_opt = 5126.497
g06_f_opt = -6961.814
g08_f_opt = -0.096
g11_f_opt = 0.750
g24_f_opt = -5.508


# TODO: x3, x4 not in objective ?!?
def g05(x1, x2, x3, x4):
    '''
    min
        f(x) =  3 * x1 + 0.000001 * x1^3 + 2*x2 + (0.000002 / 3) * x2^3
    st:
        ...
    ----------------------------
    x_opt ~= [679.945, 1026.067, 0.119, −0.396]
    f_opt ~= 5126.497
    ----------------------------
    '''
    f = 3 * x1 + 0.000001 * x1 ** 3 + 2 * x2 + (0.000002 / 3) * x2 ** 3

    g1 = np.minimum(0, -x4 + x3 - 0.55)
    g2 = np.minimum(0, -x3 + x4 - 0.55)

    h1 = 1000 * np.sin(-x3 - 0.25) + 1000 * np.sin(-x4 - 0.25) + 894.8 - x1
    h2 = 1000 * np.sin(x3 - 0.25) + 1000 * np.sin(x3 - x4 - 0.25) + 894.8 - x2
    h3 = 1000 * np.sin(x4 - 0.25) + 1000 * np.sin(x4 - x3 - 0.25) + 1294.8

    h1 = np.maximum(0, np.abs(h1) - eps)
    h2 = np.maximum(0, np.abs(h2) - eps)
    h3 = np.maximum(0, np.abs(h3) - eps)

    g3 = np.maximum(0, 0 - x1)
    g4 = np.maximum(0, x1 - 1200)
    g5 = np.maximum(0, 0 - x2)
    g6 = np.maximum(0, x2 - 1200)
    g7 = np.maximum(0, -0.55 - x3)
    g8 = np.maximum(0, x3 - 0.55)
    g9 = np.maximum(0, -0.55 - x4)
    g10 = np.maximum(0, x4 - 0.55)

    return f, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, h1, h2, h3


def g05_feasibility(x1, x2, x3, x4):
    if -x4 + x3 - 0.55 > 0:
        return False
    if -x3 + x4 - 0.55 > 0:
        return False

    if x1 < 0 or x1 > 1200:
        return False
    if x2 < 0 or x2 > 1200:
        return False
    if x3 < -0.55 or x3 > 0.55:
        return False
    if x4 < -0.55 or x4 > 0.55:
        return False

    if np.abs(1000 * np.sin(-x3 - 0.25) + 1000 * np.sin(-x4 - 0.25) + 894.8 - x1) - eps > 0:
        return False
    if np.abs(1000 * np.sin(x3 - 0.25) + 1000 * np.sin(x3 - x4 - 0.25) + 894.8 - x2) - eps > 0:
        return False
    if np.abs(1000 * np.sin(x4 - 0.25) + 1000 * np.sin(x4 - x3 - 0.25) + 1294.8) - eps > 0:
        return False

    return True


def g06(x1, x2):
    """
    min     f(x) = (x1 − 10)^3 + (x2 − 20)^3

    s.t.:
            g1(x) = −(x1 − 5)^2 − (x2 − 5)^2 + 100 ≤ 0
            g2(x) = (x1 − 6)^2 + (x2 − 5)^2 − 82.81 ≤ 0

            13 ≤ x1 ≤ 100
            0 ≤ x2 ≤ 100
    ------------------------------
    f_opt ~= −6961.814
    x_opt ~= [14.095, 0.843]
    both constraints active at optimum
    ------------------------------
    """
    f = (x1 - 10) ** 3 + (x2 - 20) ** 3

    g1 = -(x1 - 5) ** 2 - (x2 - 5) ** 2 + 100
    g2 = (x1 - 6) ** 2 + (x2 - 5) ** 2 - 82.81

    g3 = np.maximum(0, 13 - x1)
    g4 = np.maximum(0, x1 - 100)
    g5 = np.maximum(0, 0 - x2)
    g6 = np.maximum(0, x2 - 100)

    return f, np.maximum(0, g1), np.maximum(0, g2), g3, g4, g5, g6


def g06_feasibility(x1, x2):
    if -(x1 - 5) ** 2 - (x2 - 5) ** 2 + 100 > 0:
        return False
    if (x1 - 6) ** 2 + (x2 - 5) ** 2 - 82.81 > 0:
        return False

    if x1 < 13 or x1 > 100:
        return False

    if x2 < 0 or x2 > 100:
        return False

    return True


def g08(x1, x2):
    '''
    min
        f
    st:
        g1(x) = x1^2 − x2 + 1 ≤ 0
        g2(x) = 1 − x1 + (x2 − 4)^2 ≤ 0

        0 ≤ x1 ≤ 10
        0 ≤ x2 ≤ 10
    -----------------------
    f_opt ~= −0.096
    x_opt ~= [1.228, 4.245]
    -----------------------
    '''
    f = - ((np.sin(2 * np.pi * x1) ** 3) * (np.sin(2 * np.pi * x2))) / (x1 ** 3 * (x1 + x2))

    g1 = x1 ** 2 - x2 + 1
    g2 = 1 - x1 + (x2 - 4) ** 2

    g3 = np.maximum(0, 0 - x1)
    g4 = np.maximum(0, x1 - 10)
    g5 = np.maximum(0, 0 - x2)
    g6 = np.maximum(0, x2 - 10)

    return f, np.maximum(0, g1), np.maximum(0, g2), g3, g4, g5, g6


def g08_feasibility(x1, x2):
    if x1 ** 2 - x2 + 1 > 0:
        return False
    if 1 - x1 + (x2 - 4) ** 2 > 0:
        return False

    if x1 < 0 or x1 > 10:
        return False

    if x2 < 0 or x2 > 10:
        return False

    return True


def g11(x1, x2):
    '''
    min
        f(x) = x1^2 + (x2 − 1)^2
    st:
        h(x) = x2 - x1^2 = 0
        -1 =< x1 =< 1
        -1 =< x2 =< 1
    ------------------------
    x_opt ~= [−0.707, 0.500]
    f_opt ~= 0.750
    ------------------------
    '''

    f = x1 ** 2 + (x2 - 1) ** 2
    h = np.abs(x2 - x1 ** 2) - eps

    g1 = np.maximum(0, -1 - x1)
    g2 = np.maximum(0, x1 - 1)
    g3 = np.maximum(0, -1 - x2)
    g4 = np.maximum(0, x2 - 1)

    return f, np.maximum(0, h), g1, g2, g3, g4


def g11_feasibility(x1, x2):
    if np.abs(x2 - x1 ** 2) - eps > 0:
        return False

    if x1 < -1 or x1 > 1:
        return False

    if x2 < -1 or x2 > 1:
        return False

    return True


def g24(x1, x2):
    '''
    min
        f(x) = -x1 - x2
    st:
        g1(x) = -2 * x1^4 + 8 * x1^3 - 8 * x1^2 + x2 - 2  =< 0
        g2(x) = -4 * x1^4 + 32 * x1^3 - 88 * x1^2 + 96 * x1 + x2 - 36  =< 0
        0 <= x1 <= 3
        0 <= x2 <= 4

    -----------------------
    x_opt ~= [2.329  3.178]
    f_opt ~= -5.508
    -----------------------
    '''

    f = -x1 - x2
    g1 = -2 * x1 ** 4 + 8 * x1 ** 3 - 8 * x1 ** 2 + x2 - 2
    g2 = -4 * x1 ** 4 + 32 * x1 ** 3 - 88 * x1 ** 2 + 96 * x1 + x2 - 36

    g3 = np.maximum(0, 0 - x1)
    g4 = np.maximum(0, x1 - 3)
    g5 = np.maximum(0, 0 - x2)
    g6 = np.maximum(0, x2 - 4)

    return f, np.maximum(0, g1), np.maximum(0, g2), g3, g4, g5, g6


def g24_feasibility(x1, x2):
    if -2 * x1 ** 4 + 8 * x1 ** 3 - 8 * x1 ** 2 + x2 - 2 > 0:
        return False
    if -4 * x1 ** 4 + 32 * x1 ** 3 - 88 * x1 ** 2 + 96 * x1 + x2 - 36 > 0:
        return False

    if x1 < 0 or x1 > 3:
        return False

    if x2 < 0 or x2 > 4:
        return False

    return True


def main():
    # for objective functions testing
    ...


if __name__ == '__main__':
    main()
