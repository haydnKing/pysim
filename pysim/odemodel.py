from scipy.integrate import odeint
from scipy.optimize import fsolve
import numpy as np, re

from .reaction import Reaction
from .symbols import SymbolTable
from .exceptions import *

class ODEModel:
    def __init__(self, species, params, functions, reactions):
        self.species = species
        self.functions = functions
        self.params = params
        self.reactions = reactions

    @classmethod
    def fromFile(cls, filename):
        species = SymbolTable(float, 0.0)
        functions = SymbolTable(str, "0.0")
        params = SymbolTable(float, 0.0)
        reactions = []

        kw = {'species': species,
              'func': functions,
              'param': params,
              'params': params,
             }

        eq = re.compile(r'([^\s=]+)\s*(?:=\s*(.+))?')

        with open(filename) as f:
            for i,line in enumerate(f):
                line = line.strip()
                #ignore empty lines and comments
                if not line or line[0] == '#':
                    continue
                
                try:
                    #find species and params
                    syms = line.split()
                    if syms[0] in kw.keys():
                        for d in " ".join(syms[1:]).split(','):
                            m = eq.match(d.strip())
                            if not m:
                                raise SyntaxParseError(syms[0], d.strip())
                            name, value = m.groups()
                            if (name in species or
                                name in functions or
                                name in params):
                                raise DuplicateNameError(name)
                            kw[syms[0]].addSymbol(name, value)

                    else: #reaction
                        reactions.append(Reaction.fromStr(line, 
                                                          params, 
                                                          species))
                except ParseError as p:
                    p.setLine(i+1)
                    raise
        return cls(species, params, functions, reactions)

    def __str__(self):
        lines = [
            'param ' + str(self.params),
            'species ' + str(self.species),
        ]
        #if self.functions:
        #    lines += str(self.functions)

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

    def set(self, **kwargs):
        """set params or species initial conditions"""
        for k,v in kwargs.items():
            if k in self.params:
               self.params.setValueByName(k,v)
            elif k in self.species:
                self.species.setValueByName(k,v)
            else:
                raise KeyError("\"{}\" is not a known species or parameter"
                               .format(k))

    def get(self, name):
        """get params or species initial conditions"""
        if name in self.params:
           return self.params.getValueByName(name)
        elif name in self.functions:
           return self.functions.getValueByName(name)
        elif name in self.species:
           return self.species.getValueByName(name)
        else:
            raise KeyError("\"{}\" is not a known species or parameter"
                           .format(name))



    def solve(self, use_fprime=True):
        """Calculate the steady state solution of the system of equations"""
        
        fprime = None
        if use_fprime:
            fprime = self._get_fprime()

        out = fsolve(self._get_f(), 
                     self.species.values,
                     fprime=fprime,
                     col_deriv=False)

        return out



