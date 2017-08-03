
from context import pysim

import unittest, os.path, numpy as np, numpy.testing as npt

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

class RateTests(unittest.TestCase):
    def setUp(self):
         self.model = pysim.ODEModel.fromFile(os.path.join(test_data, 
                                                           "ratetest.model"))

    def test_fw_reaction(self):
        fn = self.model.reactions[0].getRateEquation()
        exp = np.array([-1.0, 1.0, 0.0])
        tests = [([1,1,1], 1.0*exp),
                 ([1,2,2], 1.0*exp),
                 ([7,2,2], 7.0*exp),]
        self._helper(fn, tests)

    def test_reversible_reaction(self):
        fn = self.model.reactions[1].getRateEquation()
        exp = np.array([-1.0, 1.0, 0.0])
        tests = [([1,1,1],  0.0*exp),
                 ([1,2,2], -1.0*exp),
                 ([7,2,2],  5.0*exp),]
        self._helper(fn, tests)

    def test_MM_reaction(self):
        fn = self.model.reactions[2].getRateEquation()
        exp = np.array([-1.0, 1.0, 0.0])
        tests = [([1,1,1], 0.5*exp),
                 ([1,2,2], 1.0*exp),
                 ([7,2,2], 2.8*exp),]
        self._helper(fn, tests)

    def _helper(self, fn, tests):
        for i,(y, resp) in enumerate(tests):
            npt.assert_allclose(fn(y), resp, err_msg="case {}".format(i))



if __name__ == "__main__":
    unittest.main()
