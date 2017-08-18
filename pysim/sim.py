import numpy as np, pandas as pd, itertools

from .odemodel import ODEModel

class Sim:
    """Test a model under a variety of conditions"""
    def __init__(self, model):
        self.model = model
        self.defaults = model.params.merge(model.consts)
        self.query = None

    @classmethod
    def fromFile(self, modelfile):
        model = ODEModel.fromFile(modelfile)
        return Sim(model)

    def numParameters(self):
        return len(self.defaults)

    def addSweep(self, sweeps, relative=False):
        """
            Add a parameter sweep to the simulation
            sweeps: dictionary of parameter names mapping to values to set
                all possibile combinations will be solved.
                Names should be pure parameter names or simple ratios of 
                parameters, such as "a / b", in which case the ratio a/b will
                be varied with the product ab held constant.
                All names in sweeps should be independend, setting a sweep
                on "a / b" and "b / c" will generate an error, instead try
                setting a separate sweep.
            relative: if true, sweep values are multiplied by the default value
        """
        #check each name only features once
        sweepnames = []
        for name in sweeps.keys():
            names = [n.strip() for n in name.split('/')]
            for n in names:
                if n in sweepnames:
                    raise ValueError(("Name \'{}\' used multiple times in " +
                                      "argument to addSweep").format(n))

        sweeplist = [[x,] for x in self.defaults.values]
        ratio_idx    = []
        ratio_prod   = []
        ratio_values = []
        for k,v in sweeps.items():
            if '/' in k:
                names = [n.strip() for n in k.split('/')]
                if len(names) != 2:
                    raise ValueError("Invalid ratio spec \'{}\'".format(k))
                idx = [self.defaults.getIndexByName(n) for n in names]
                ratio_idx.append(idx)
                ratio_prod.append(np.product(
                    [self.defaults.values[i] for i in idx]))
                if relative:
                    ratio_values.append(v * 
                                        self.defaults.values[idx[0]] / 
                                        self.defaults.values[idx[1]])
                else:
                    ratio_values.append(v)
            else:
                idx = self.defaults.getIndexByName(k)
                if relative:
                    sweeplist[idx] = np.array(v) * sweeplist[idx]
                else:
                    sweeplist[idx] = np.array(v)

        rows = np.product([len(x) for x in sweeplist + ratio_values]) 

        data = np.zeros((rows, self.numParameters()))

        lparams = len(self.defaults)
        for i,p in enumerate(itertools.product(*(sweeplist+ratio_values))):
            row = list(p[:lparams])
            #set the ratio parameters
            for j,r in enumerate(p[lparams:]):
                a = np.sqrt(r * ratio_prod[j])
                row[ratio_idx[j][0]] = a
                row[ratio_idx[j][1]] = ratio_prod[j] / a

            data[i] = row

        if self.query:
            self.query = np.append(self.query, data, axis=0)
        else:
            self.query = data

    def setDefault(self, key, value):
        self.model.set(key, value)

    def setRatio(self, numerator, denomenator, ratio, product=1.0):
        P = np.sqrt(product)
        self.model.set(numerator, P * np.sqrt(ratio))
        self.model.set(denomenator, P/np.sqrt(ratio))

    def addQuery(self, query):
        if query.shape[2] != self.numParameters():
            raise ValueError("Queries need {} parameters, not {}".format(
                self.numParameters(), query.shape[2]))
        if self.query:
            self.query = np.append(self.query, query, axis=0)
        else:
            self.query = query.copy()

    def solve(self, use_jacobian=True):

        print("kp = {}".format(self.model.get('kp')))
        print("mu = {}".format(self.model.get('mu')))

        o = np.zeros((self.query.shape[0], len(self.model.species)))
        last_solved = -1
        n_solved = 0
        for i in range(o.shape[0]):
            print("{} / {} / {}       ".format(i, 
                                               n_solved, 
                                               o.shape[0]), end="\r")
            o[i,:], solved = self.model.solveForParams( 
                self.query[i,:], 
                o[last_solved,:] if last_solved > 0 else None,
                use_jacobian)
            if solved:
                n_solved += 1
                last_solved = i

        print("Solved {} of {}".format(n_solved, o.shape[0]))

        df = pd.DataFrame(
            data=np.append(self.query,o,1),
            columns=(self.model.params.names + 
                     self.model.consts.names +
                     self.model.species.names))

        return df


