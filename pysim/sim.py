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

    def sweep(self, parameters=[], ratios=[]):
        """
            Setup a sweep experiment, where one or many parameters are 
            varies over a range of values
            Each parameter can only be used once, so you can't sweep over 
            parameter A and the ratio A/B, for example. 
            parameters: a list of tuples where each tuple is
                (parameter_name, list_or_array_of_values_to_test)
            ratios: a list of tuples specifying ratios to scan as
                    (numerator_name, 
                    denomenator_name, 
                    ratios_to_test <list or array>,
                    products_to_test <list or array, optional>)
                Ratios sets a/b, products sets a.b.
                If missing, the list of products will be set to one.
            Calling this function resets any previous sweeps set.
        """
        #check names are unique
        names = []
        for n in ([p[0] for p in parameters] + 
                  [r[i] for i in [0,1] for r in ratios]):
            if n in names:
                raise ValueError("Can't sweep on {} more than once".format(n))
            names.append(n)

        symbols = self.model.params.merge(self.model.consts)

        toiter = lambda v: v if hasattr(v, '__iter__') else [v,]

        #convert everything to arrays and indexes
        parameters = [(symbols.getIndexByName(p[0]), 
                       np.array(toiter(p[1]))) for p in parameters]
        ratios = [(r[0], r[1], r[2], 1.0,) if len(r) == 3 else r
                    for r in ratios]
        ratios = [(symbols.getIndexByName(r[0]), 
                   symbols.getIndexByName(r[1]), 
                   np.array(toiter(r[2])), 
                   np.array(toiter(r[3])),) 
                    for r in ratios]

        #get the indexes for each sweep
        sweep = []
        for p in parameters:
            sweep.append(range(len(p[1])))
        for r in ratios:
            sweep.append(list(itertools.product(range(len(r[2])),
                                                range(len(r[3])))))
        rows = np.product([len(s) for s in sweep])

        data = np.zeros((rows, len(symbols)))

        P = len(parameters)
        R = len(ratios)

        for row, vals in enumerate(itertools.product(*sweep)):
            o = symbols.values.copy()
            for p,val in zip(parameters, vals[:P]):
                o[p[0]] = p[1][val]    
            for r,val in zip(ratios, vals[P:]):
                o[r[0]] = np.sqrt(r[3][val[1]] * r[2][val[0]])            
                o[r[1]] = np.sqrt(r[3][val[1]] / r[2][val[0]])            

            data[row,:] = o

        self.query = data

    def setDefault(self, key, value):
        self.model.set(key, value)

    def getDefault(self, key):
        return self.model.get(key)

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


