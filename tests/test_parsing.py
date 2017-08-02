
from context import pysim

import unittest, os.path

test_data = os.path.join(os.path.dirname(__file__), "data/")

class ParsingTestSuite(unittest.TestCase):
    """Test parsing of model definitions"""

    successes = ['simple']

    def test_success(self):
        for s in self.successes:
            model = pysim.ODEModel.fromFile(
                        os.path.join(test_data, "{}.test.model".format(s)))
            with open(os.path.join(test_data, 
                                        "{}.expected.model".format(s))) as f:
                expected = f.read()
            self.assertEqual(expected, str(model))

if __name__ == "__main__":
    unittest.main()
