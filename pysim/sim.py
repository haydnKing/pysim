import numpy as np, pandas as pd, itertools

from .odemodel import ODEModel

class Sim:
    """Test a model under a variety of conditions"""
    def __init__(self, model):
        self.model = model
        self.defaults = model.params.copy()
        self.sweeps = [None, ] * len(self.defaults)

    @classmethod
    def fromFile(self, modelfile):
        model = ODEModel.fromFile(modelfile)
        return Sim(model)

    def setAbsSweep(self, name, values):
        """name: parameter name to vary
            values: np.array of values to set parameter during sweep"""
        idx = self.defaults.getIndexByName(name)
        self.sweeps[idx] = values

    def setRelSweep(self, name, values):
        """Set Abs sweep, but relative to default value instead"""
        self.setAbsSweep(name, values * self.defaults.getValueByName(name))

    def _build_query_array(self):
        """build the output dataframe"""
        sweeps = [x if x is not None else [self.defaults.values[i],] 
                  for i,x in enumerate(self.sweeps)]
        rows = np.product([len(x) for x in sweeps]) 

        data = np.zeros((np.product([len(x) for x in sweeps]), 
                         len(self.defaults)))

        for i,r in enumerate(itertools.product(*sweeps)):
            data[i] = r

        return data

    @staticmethod
    def _solve(model, args, output):
        for i in range(len(args)):
            d = {}
            for k,v in zip(model.params.names, args[i]):
                d[k] = v
            model.set(**d)
            output[i] = model.solve()

    def solve(self):
        q = self._build_query_array()
        o = np.zeros((len(q), len(self.model.species)))
        self._solve(self.model, q, o)

        df = pd.DataFrame(
            data=np.append(q,o,1),
            columns=self.model.params.names + self.model.species.names)

        return df


