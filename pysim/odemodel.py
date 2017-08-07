from scipy.integrate import odeint
from scipy.optimize import fsolve
import numpy as np

from .reaction import Reaction
from .symbols import SymbolTable
from .exceptions import ParseError

class ODEModel:
    def __init__(self, species, params, reactions):
        self.species = species
        self.params = params
        self.reactions = reactions

    @classmethod
    def fromFile(cls, filename):
        species = SymbolTable()
        params = SymbolTable()
        reactions = []
        with open(filename) as f:
            for i,line in enumerate(f):
                line = line.strip()
                #ignore empty lines and comments
                if not line or line[0] == '#':
                    continue
                
                try:
                    #find species and params
                    syms = line.split()
                    if syms[0] == "species":
                        for declaration in " ".join(syms[1:]).split(','):
                            name = declaration.split('=')[0]
                            if name in params:
                                raise DuplicateNameError(name)
                            species.addFromStr(declaration)
                    elif syms[0] == "param":
                        for declaration in " ".join(syms[1:]).split(','):
                            name = declaration.split('=')[0]
                            if name in species:
                                raise DuplicateNameError(name)
                            params.addFromStr(declaration)
                    else: #reaction
                        reactions.append(Reaction.fromStr(line, 
                                                          params, 
                                                          species))
                except ParseError as p:
                    p.setLine(i+1)
                    raise
        return cls(species, params, reactions)

    def __str__(self):
        lines = [
            'param ' + str(self.params),
            'species ' + str(self.species),
        ]

        lines += [str(r) for r in self.reactions]
        return '\n'.join(lines) + '\n'

    def _get_f(self):
        #stoichiometry matrix len(species)xlen(reactions)
        S = np.array([r.getStoiciometry() for r in self.reactions]).T
        rates = [r.getRateEquation() for r in self.reactions]
        def f(y):
            ret = np.array([r(y) for r in rates])
            return S.dot(ret)
        return f

    def _get_fprime(self):
        #stoichiometry matrix len(species)xlen(reactions)
        S = np.array([r.getStoiciometry() for r in self.reactions]).T
        J = [r.getJacobianEquation() for r in self.reactions]
        def j(y):
            _j = [r(y) for r in J]
            return S.dot(np.array([r(y) for r in J]))
        return j

    def solve(self, use_fprime=True):
        
        fprime = None
        if use_fprime:
            fprime = self._get_fprime()

        out = fsolve(self._get_f(), 
                     self.species.values,
                     fprime=fprime,
                     col_deriv=False)

        return out



