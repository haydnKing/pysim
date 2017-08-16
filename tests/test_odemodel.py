
from context import pysim

import unittest, os.path, numpy as np, numpy.testing as npt
import scipy.optimize

test_data = os.path.join(os.path.dirname(__file__), "data/")

class ParsingTestSuite(unittest.TestCase):
    """Test parsing of model definitions"""

    successes = ['1', '2', '3', '4']

    def test_success(self):
        for s in self.successes:
            model = pysim.ODEModel.fromFile(
                        os.path.join(test_data, "{}.test.model".format(s)))
            with open(os.path.join(test_data, 
                                        "{}.expected.model".format(s))) as f:
                expected = f.read()
            self.assertEqual(expected, str(model), "in test {}".format(s))

class ReactionRateTests(unittest.TestCase):
    def setUp(self):
         self.model = pysim.ODEModel.fromFile(os.path.join(test_data, 
                                                           "ratetest.model"))

    def test_fw_reaction(self):
        fn = self.model.reactions[0].getRateEquation()
        tests = [([1,1,1], 1.0),
                 ([1,2,2], 1.0),
                 ([7,2,2], 7.0),]
        self._helper(fn, tests)

    def test_reversible_reaction(self):
        fn = self.model.reactions[1].getRateEquation()
        tests = [([1,1,1],  0.0),
                 ([1,2,2], -1.0),
                 ([7,2,2],  5.0),]
        self._helper(fn, tests)

    def test_MM_reaction(self):
        fn = self.model.reactions[2].getRateEquation()
        tests = [([1,1,1], 0.5),
                 ([1,2,2], 1.0),
                 ([7,2,2], 2.8),]
        self._helper(fn, tests)

    def _helper(self, fn, tests):
        for i,(y, resp) in enumerate(tests):
            self.assertEqual(fn(y), resp, "case {}".format(i))

class GoverningFunctionTests(unittest.TestCase):
    def setUp(self):
         self.model = pysim.ODEModel.fromFile(os.path.join(test_data, 
                                                           "ratetest.model"))

    def test_unwrapped(self):
        tests = [np.array([1.0,1.0,1.0]),
                 np.array([2.0,1.0,1.0]),
                 np.array([1.0,2.0,0.0]),
                 np.array([1.0,1.0,2.0]),]
        f = self.model._get_unwrapped_f()
        for q in tests:
            npt.assert_allclose(f(q), 
                                self._rate(q), 
                                atol=10e-12,
                                err_msg="q = {}".format(q))

    def test_wrapped(self):
        tests = [np.array([1.0,1.0,1.0]),
                 np.array([2.0,1.0,1.0]),
                 np.array([1.0,-2.0,0.0]),
                 np.array([1.0,1.0,2.0]),]
        f = self.model._get_f()
        for q in tests:
            npt.assert_allclose(f(q), 
                                self._rate(np.square(q))/(2*q), 
                                atol=10e-12,
                                err_msg="q = {}".format(q))

    def _rate(self, q):
        kfw = self.model.get("k_fw")
        krv = self.model.get("k_rv")
        kcat = self.model.get("k_cat")
        km = self.model.get("k_m")

        a = q[1] * krv - 2 * kfw * q[0] - kcat*q[2]*q[0] / (km + q[0])
        return np.array([a, -a, 0])

class NumericalJacobianTest(unittest.TestCase):
    def setUp(self):
        self.load_model("jacobiantest.model")

    def test_jac(self):

        points = [[1.5, 1.5],
                  [1.5, 1.0],
                  [27.0, -12.],
                  [-1.5, 1.5],
                  [0., 1.],
                 ]

        self.check_points(points)


    def load_model(self, name):
        self.model = pysim.ODEModel.fromFile(os.path.join(test_data, name))
        self.f = self.model._get_f()
        self.J = self.model._get_fprime()


    def check_points(self, points):
        for z in points:
            J_num = self.set_inf(self.calc_J(np.array(z)))
            J_eval= self.set_inf(self.J(np.array(z)))

            npt.assert_allclose(J_eval, 
                                J_num,
                                rtol=1e-4,
                                atol=1e-6,
                                err_msg="at point {}".format(z))

    def set_inf(self, mat, max_val = 1e10):
        """Assume all numbers greater than max_val are equal"""
        for m in range(mat.shape[0]):
            for n in range(mat.shape[1]):
                if np.abs(mat[m,n]) > max_val:
                    mat[m,n] = np.sign(mat[m,n]) * max_val

        return mat

    def calc_J(self, z):
        eps = np.sqrt(np.finfo(float).eps)

        funcs = lambda i: (lambda z: self.f(z)[i])

        J = np.array([scipy.optimize.approx_fprime(z, funcs(i), eps) 
                      for i in range(len(z))])

        return J



class MoreJacobianTests(NumericalJacobianTest):
    def setUp(self):
        self.load_model("solvetest1.model")

    def test_jac(self):

        points = [[1., 1.],
                  [5., 1.],
                  [0., 1.],
                  [0.25, 17.],
                 ]

        self.check_points(points)


class MMJacobianTests(NumericalJacobianTest):
    def setUp(self):
        self.load_model("MMJacobiantest.model")

    def test_jac(self):

        points = [[1., 1., 1.],
                  [0., 0., 0.],
                  [1., 0., 5.],
                  [0.25, 17., 5.2],
                 ]

        self.check_points(points)


class SolveTests(unittest.TestCase):
    def setUp(self):
        self.model = pysim.ODEModel.fromFile(os.path.join(test_data, 
                                                          "solvetest1.model"))
         
    def _solution(self):
        k_r = self.model.get("k_r")
        k_d = self.model.get("k_d")
        d = self.model.get("d")

        x = 0.25 * (-d + np.sqrt(d*d+8.*k_d*k_r)) / k_d
        y = x*x*k_d / d

        return np.array([x,y])


    def test_solve_without_J(self):
        out = self.model.solve(False)
        npt.assert_allclose(out, self._solution())

    def test_solve_with_J(self):
        out = self.model.solve(True)
        npt.assert_allclose(out, self._solution())

class SolveTests2(unittest.TestCase):
    def setUp(self):
        self.model = pysim.ODEModel.fromFile(os.path.join(test_data, 
                                                          "solvetest2.model"))
         
    def _solution(self):
        kp = self.model.get("kp")
        kb = self.model.get("kb")
        mu = self.model.get("mu")

        #find r
        a = kb
        b = kb + mu
        c = -kp
        
        r = (-b + np.sqrt(b*b-4*a*c))/ (2.*a)
        p = 1. / (1 + r * kb / mu)

        return np.array([p,r])


    def test_solve_without_J(self):
        out = self.model.solve(False)
        npt.assert_allclose(out, self._solution())

    def test_solve_with_J(self):
        out = self.model.solve(True)
        npt.assert_allclose(out, self._solution())

class SetTests(SolveTests):

    def test_setSolve(self):
        self.model.set(k_d = 7., d=2.5, k_r=3.)
        npt.assert_allclose(self.model.solve(), self._solution())

    def test_badSet(self):
        self.assertRaises(KeyError, self.model.set, not_a_symbol=7.)

class MMSolveTests(unittest.TestCase):
    def setUp(self):
        self.model = pysim.ODEModel.fromFile(os.path.join(test_data, 
                                                           "MMJacobiantest.model"))
        k1 = self.model.params.getValueByName("k1")
        k2 = self.model.params.getValueByName("k2")
        kc = self.model.params.getValueByName("k_cat")
        km = self.model.params.getValueByName("k_m")

        self.solution = np.array([km/(kc/k2-1), k1/k2, k1/k2])

    def test_solve_without_J(self):
        out = self.model.solve(False)
        npt.assert_allclose(out, self.solution)

    def test_solve_with_J(self):
        out = self.model.solve(True)
        npt.assert_allclose(out, self.solution)

if __name__ == "__main__":
    unittest.main()
