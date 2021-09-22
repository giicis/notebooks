from pymoo_rpp_gen import ReleasePlanningProblem
import unittest
import yaml
import numpy as np


def nvars() -> int:
    data_file = "/home/leo/workspace/giicis/notebooks/datasets/rpp_data_soft.yaml"
    with open(data_file, 'r') as f:
        rpp = yaml.safe_load(f.read())
    return 1 + rpp['number_of_requirements'] + rpp['number_of_releases'] * rpp['number_of_requirements']


def instace_problem():
    data_file = "/home/leo/workspace/giicis/notebooks/datasets/rpp_data_soft.yaml"
    with open(data_file, 'r') as f:
        model = yaml.safe_load(f.read())
    problem = ReleasePlanningProblem(model)
    return problem


class TestPymooRppGen(unittest.TestCase):
    def test_get_xs(self):
        problem = instace_problem()
        nvar = nvars()

        expected_x_shape = (5,)
        xs = problem._get_xs(np.zeros(nvar))
        self.assertEqual(xs.shape, expected_x_shape)

    def test_get_ys(self):
        problem = instace_problem()
        nvar = nvars()

        expected_y_shape = (3, 5)
        ys = problem._get_ys(np.zeros(nvar))
        self.assertEqual(ys.shape, expected_y_shape)


if __name__ == "__main__":
    unittest.main()
