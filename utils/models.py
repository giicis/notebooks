import pyomo.environ as pyo
import numpy as np


def __rpp_soft_abstract_model():
    """
    Creates an abstract model of Rpp problem with soft restrictions
    """
    rpp = pyo.AbstractModel()
    rpp.name = "Release planning problem with soft restrictions"

    # Model's parameters
    rpp.number_of_requirements = pyo.Param(within=pyo.NonNegativeIntegers)
    rpp.number_of_stakeholders = pyo.Param(within=pyo.NonNegativeIntegers)
    rpp.number_of_releases = pyo.Param(within=pyo.NonNegativeIntegers)
    rpp.pm = pyo.Param(within=pyo.NonNegativeIntegers)
    rpp.pd = pyo.Param(within=pyo.NonNegativeIntegers)
    rpp.f0 = pyo.Param(within=pyo.Reals)
    rpp.f1 = pyo.Param(within=pyo.Reals)

    # Sets that will be used to iterate over
    rpp.requirements = pyo.RangeSet(1, rpp.number_of_requirements)
    rpp.stakeholders = pyo.RangeSet(1, rpp.number_of_stakeholders)
    rpp.releases = pyo.RangeSet(1, rpp.number_of_releases)

    # Parameters defined over previous defined sets
    rpp.efforts = pyo.Param(rpp.requirements)
    rpp.profit = pyo.Param(rpp.stakeholders)

    # Relations defined over the cartesian product of sets
    # (i,j) requierement i should be implemented if j is implemented
    rpp.precedence = pyo.Set(within=rpp.requirements * rpp.requirements)
    # (s,i) > 0 if stakeholder s has interest over requierement i
    # This relation is here beacuse the dataset have this information
    # We are using this to initialize matrix A
    rpp.interest = pyo.Set(within=rpp.stakeholders * rpp.requirements)

    # We use this function to assign a requierement priority for each stakeholder
    # This is because the dataset we are using does not have this information
    def A_init(rpp, s, i):
        if (s, i) in rpp.interest:
            return 1
        return 0
    # This parameter needs to be mutable so later on we can normalize it
    rpp.A = pyo.Param(rpp.stakeholders, rpp.requirements,
                      initialize=A_init, mutable=True)

    # Variables
    # Used in wernes approach
    rpp.a = pyo.Var(bounds=(0, 1))
    # Store the number in which the requierement is implemented
    rpp.x = pyo.Var(rpp.requirements, domain=pyo.NonNegativeIntegers)
    # y[l,i] == 1 if requierement i is implemented in l release
    rpp.y = pyo.Var(rpp.releases, rpp.requirements, domain=pyo.Binary)

    # Objetive function
    def obj_function_rule(rpp):
        return rpp.a
    rpp.OBJ = pyo.Objective(rule=obj_function_rule, sense=pyo.maximize)

    # Constraints
    def profit_soft_constraint_rule(rpp):
        def inner_sum(s): return sum(
            rpp.A[s, i] * (rpp.number_of_releases - rpp.x[i]) for i in rpp.requirements)
        return sum(rpp.profit[s] * inner_sum(s) for s in rpp.stakeholders) >= rpp.f1 + rpp.a * (rpp.f0 - rpp.f1)
    rpp.profit_soft_constraint = pyo.Constraint(
        rule=profit_soft_constraint_rule)

    def release_constraint_rule(rpp, i):
        return sum(rpp.y[l, i] * l for l in rpp.releases) == rpp.x[i]
    rpp.release_constraint = pyo.Constraint(
        rpp.requirements, rule=release_constraint_rule)

    def implementation_constraint_rule(rpp, i):
        return sum(rpp.y[l, i] for l in rpp.releases) == 1
    rpp.implementation_constraint = pyo.Constraint(
        rpp.requirements, rule=implementation_constraint_rule)

    def effort_soft_constraint_rule(rpp, l):
        return sum(rpp.efforts[i] * rpp.y[l, i] for i in rpp.requirements) <= rpp.pm - rpp.a * (rpp.pm - rpp.pd)
    rpp.efforts_soft_constraint = pyo.Constraint(pyo.RangeSet(
        1, rpp.number_of_releases - 1), rule=effort_soft_constraint_rule)

    def precende_constraint_rule(rpp, i, j):
        return rpp.x[i] <= rpp.x[j]
    rpp.precedence_constraint = pyo.Constraint(
        rpp.precedence, rule=precende_constraint_rule)

    return rpp


def __normalize(x):
    """
    Given a numpy vector X, returns another vector the where the sum of elements is 1
    """
    acc = np.sum(x)
    if acc == 0:
        return x
    return x / acc


def __A_normalizate(rpp):
    """
    Given an rpp model with A matrix
    Normalize each row, so the sum of elements per row is 1
    """
    A = np.zeros((rpp.number_of_stakeholders.value,
                 rpp.number_of_requirements.value))

    # Assing rpp.A values to A
    for (i, j) in rpp.A.index_set():
        A[i - 1, j - 1] = rpp.A[i, j].value

    # Normalize A
    for i in range(0, A.shape[0]):
        A[i, :] = __normalize(A[i, :])

    # Assign A values to rpp.A
    for j in range(0, A.shape[1]):
        for i in range(0, A.shape[0]):
            rpp.A[i + 1, j + 1] = A[i, j]


def __construct_rpp_soft(data_file: str) -> pyo.ConcreteModel:

    rpp_soft = __rpp_soft_abstract_model()
    rpp_concrete = rpp_soft.create_instance(data=data_file)
    __A_normalizate(rpp_concrete)

    return rpp_concrete


def get_instance(name: str, data_file: str) -> pyo.ConcreteModel:
    if name == 'rpp_soft':
        return __construct_rpp_soft(data_file)
    else:
        raise NotImplementedError
