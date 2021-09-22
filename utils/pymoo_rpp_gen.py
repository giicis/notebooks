from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.model.problem import Problem
from pymoo.algorithms.so_genetic_algorithm import GA
import numpy as np
import yaml


class ReleasePlanningProblem(Problem):

    def get_mask(self) -> "list[str]":
        rpp = self._model
        mask = ['real']
        xs = ['int' for i in range(rpp['number_of_requirements'])]
        ys = ['bin' for i in range(
            rpp['number_of_requirements'] * rpp['number_of_releases'])]
        return mask + xs + ys

    def nconstraints(self) -> int:
        rpp = self._model
        nprofit = 1
        # *2 is here because this constraints are == but pymoo requires them to be <=
        # So we add another constraint to transfor it to <=.
        # Ej
        # x1 + x2 == 1
        # This is equal to
        # x1 + x2 >= 1
        # x1 + x2 <= 1
        # Equal to
        # -(x1 + x2) <= -1
        # x1 + x2 <= 1
        nrelease = 2 * rpp['number_of_requirements']
        nimpl = 2 * rpp['number_of_requirements']
        neffort = rpp['number_of_releases'] - 1
        nprecedence = len(rpp['precedence'])
        return nprofit + nrelease + nimpl + neffort + nprecedence

    def nvars(self) -> int:
        rpp = self._model
        # x0 is alpha
        # x1 to xn are the X in the original paper
        # xn+1 to xm are the Y decision variables stored in row major order
        return 1 + rpp['number_of_requirements'] + rpp['number_of_releases'] * rpp['number_of_requirements']

    def _get_bounds(self) -> tuple:
        rpp = self._model
        xl = [0.0 for i in range(self.nvars())]

        xu = [1.0]
        # In the original paper, X are between [0,k] being K the number of releases
        xu += [rpp['number_of_releases']
               for i in range(rpp['number_of_requirements'])]
        # This ones are binary
        xu += [1.0 for i in range(rpp['number_of_requirements']
                                  * rpp['number_of_releases'])]
        return np.array(xl), np.array(xu)

    def _get_params(self) -> tuple:
        n_var = self.nvars()
        n_constr = self.nconstraints()

        xl, xu = self._get_bounds()
        return n_var, n_constr, xl, xu

    def _normalize_and_store_matrix(self, matrix: np.array):
        for i in range(matrix.shape[0]):
            sum = np.sum(matrix[i, :])
            if sum != 0:
                matrix[i, :] = matrix[i, :] / sum
        self._matrixA = matrix

    def _generate_matrix_A(self):
        rpp = self._model
        nstakeholders = rpp['number_of_stakeholders']
        nrequirements = rpp['number_of_requirements']
        matrix = np.zeros((nstakeholders, nrequirements))
        for i, j in rpp['interest']:
            matrix[i-1, j-1] = 1
        self._normalize_and_store_matrix(matrix)

    def __init__(self, model: dict):
        self._model = model
        n_var, n_constr, xl, xu = self._get_params()
        self._generate_matrix_A()
        super().__init__(n_var=n_var, n_obj=1, n_constr=n_constr,
                         xl=xl, xu=xu, elementwise_evaluation=True)

    def _get_xs(self, X: np.array) -> np.array:
        a, b = 1, self._model['number_of_requirements'] + 1
        return X[a:b]

    def _get_ys(self, X: np.array) -> "tuple[np.array, int, int]":
        rpp = self._model
        start = rpp['number_of_requirements'] + 1
        k = rpp['number_of_releases']
        n = rpp['number_of_requirements']
        return np.reshape(X[start:], (k, n), order='C')

    def _profit_soft_constraint(self, X: np.array) -> np.array:
        alpha = X[0]
        rpp = self._model
        xs = self._get_xs(X)
        bs = np.array([rpp['profit'][idx] for idx in rpp['profit']])

        # Calculate coefficente to normalize the constraint
        #ones = np.ones(len(xs))
        #inner = np.matmul(self._matrixA, (rpp['number_of_releases'] - ones))
        #lhs = np.dot(inner, bs)
        #coef = lhs - rpp['f1'] + rpp['f0'] - rpp['f1']

        inner = np.matmul(self._matrixA, (rpp['number_of_releases'] - xs))
        lhs = np.dot(inner, bs)
        total = lhs - (rpp['f1'] + alpha * (rpp['f0'] - rpp['f1']))
        total = (-1) * total
        #total = total / coef

        return np.array([total])

    def _release_constraint(self, X: np.array) -> np.array:
        rpp = self._model
        releases = np.array(
            [i for i in range(1, rpp['number_of_releases'] + 1)])
        Y = self._get_ys(X)
        x = self._get_xs(X)
        summ = np.matmul(releases, Y)

        constr1 = x - summ
        constr2 = summ - x
        return np.concatenate((constr1, constr2))

    def _implementation_constraint(self, X: np.array) -> np.array:
        rpp = self._model
        Y = self._get_ys(X)
        k = rpp['number_of_releases']

        ones = np.ones((1, k))
        summ = np.matmul(ones, Y)
        constr1 = summ - 1
        constr2 = 1 - summ
        return np.concatenate((constr1[0], constr2[0]))

    def _effort_soft_constraint(self, X: np.array) -> np.array:
        rpp = self._model
        efforts = np.array([rpp['efforts'][key] for key in rpp['efforts']])
        Y = self._get_ys(X)
        pm = rpp['pm']
        pd = rpp['pd']
        alpha = X[0]

        summ = np.matmul(Y, efforts)
        rhs = pm - alpha * (pm - pd)
        ret = summ - rhs
        return ret

    def _precedence_constraint(self, X: np.array) -> np.array:
        rpp = self._model
        interest = rpp['interest']
        x = X
        constraint = []
        for i, j in interest:
            constraint.append(x[i] - x[j])
        return np.array(constraint)

    def _evaluate(self, X, out, *args, **kwargs):
        alpha = X[0]
        prof_c = self._profit_soft_constraint(X)
        rel_c = self._release_constraint(X)
        impl_c = self._implementation_constraint(X)
        ef_c = self._effort_soft_constraint(X)
        pre_c = self._precedence_constraint(X)

        ##print(f"prof_c: {prof_c}")
        ##print(f"rel_c: {rel_c}")
        ##print(f"ef_c: {ef_c}")
        ##print(f"pre_c: {pre_c}")
        ##print(f"impl_c: {impl_c}")
        out["F"] = alpha
        out["G"] = np.concatenate((prof_c, rel_c, ef_c, pre_c, impl_c))


def main():
    data_file = "/home/leo/workspace/giicis/notebooks/datasets/rpp_data_soft.yaml"
    with open(data_file, 'r') as f:
        model = yaml.safe_load(f.read())

    problem = ReleasePlanningProblem(model)
    mask = problem.get_mask()
    sampling = MixedVariableSampling(mask, {
        "bin": get_sampling("bin_random"),
        "int": get_sampling("int_random"),
        "real": get_sampling("real_random"),
    })

    crossover = MixedVariableCrossover(mask, {
        "bin": get_crossover("bin_ux", prob=.9),
        "int": get_crossover("int_sbx", prob=1.0, eta=3.0),
        "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    })

    mutation = MixedVariableMutation(mask, {
        "bin": get_mutation("bin_bitflip"),  # bin_bitflip
        "int": get_mutation("int_pm", eta=3.0),
        "real": get_mutation("real_pm", eta=3.0),
    })

    algorithm = GA(
        pop_size=30,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=False,
    )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 200),
        seed=0,
        save_history=True,
        verbose=True
    )

    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)

    return 0


if __name__ == "__main__":
    main()
