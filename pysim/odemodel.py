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
        species = SymbolTable(float, 1.0)
        functions = SymbolTable(str, "0.0")
        params = SymbolTable(float, 0.0)
        reactions = []

        kw = {'species': species,
              'func': functions,
              'param': params,
              'params': params,
             }

        eq = re.compile(r'([^\s=]+)\s*(?:=\s*(.+))?')

        seen_reaction = False

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
                        if syms[0] == "species" and seen_reaction:
                            raise ParseError("Cannot define species after reaction")
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
                        seen_reaction = True
                        reactions.append(
                            Reaction.fromStr(line, 
                                             params, 
                                             species.merge(functions)))
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

    def _get_unwrapped_f(self):
        #stoichiometry matrix len(species)xlen(reactions)
        S = np.array([r.getStoiciometry() for r in self.reactions]).T
        rates = [r.getRateEquation(self.species.names,
                                   self.functions) 
                 for r in self.reactions]
        def f(y):
            ret = np.array([r(y) for r in rates])
            return S.dot(ret)[:len(self.species)]
        return f

    def _get_f(self):
        f = self._get_unwrapped_f()

        def g(z):
            return f(np.square(z))
        return g

    def _get_unwrapped_fprime(self):
        #stoichiometry matrix len(species)xlen(reactions)
        S = np.array([r.getStoiciometry() for r in self.reactions]).T
        J = [r.getJacobianEquation(self.species.names,
                                   self.functions) 
                  for r in self.reactions]
        def j(y):
            return S.dot(np.array([r(y) for r in J]))[:len(self.species),
                                                      :len(self.species)]
        return j
    
    def _get_fprime(self):
        fp = self._get_unwrapped_fprime()
        f = self._get_unwrapped_f()
        eps = np.sqrt(np.finfo(float).eps)
        eps_v = [np.zeros(len(self.species)) for i in range(len(self.species))]
        for i in range(len(self.species)):
            eps_v[i][i] = eps

        def g(z):
            x = np.square(z)
            R = fp(x)
            for i in range(len(z)):
                R[:,i] *= 2 * z[i]
            #F = f(x)
            #for n in range(R.shape[0]):
            #    for m in range(R.shape[1]):
            #        if z[n] != 0.:
            #            s = F[m] / (2 * z[n] * z[n])
            #        else:
            #            P = f(x + eps_v[m])[m]
            #            if np.abs(P) < 1:
            #                s = 0.
            #            elif P > 0:
            #                s = np.finfo(float).max
            #            else:
            #                s = np.finfo(float).min
            #        R[n,m] -= s  

            return R

        return g

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

    def setAllParams(self, params):
        """Set all parameters"""
        if len(self.params) != len(params):
            raise ValueError("Expected {} parameters, got {}".format(
                len(self.params), len(params)))
        self.params.values = np.array(params)

    def setAllSpecies(self, species):
        """Set all initial conditions"""
        if len(self.species) != len(species):
            raise ValueError("Expected {} species, got {}".format(
                len(self.species), len(species)))
        self.species.values = np.array(species)

    def solveForParams(self, params, initial_conditions=None, use_fprime=True):
        self.setAllParams(params)
        if initial_conditions is not None:
            self.setAllSpecies(initial_conditions)
        return self.solve(use_fprime)

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

        (out, info, ier, mesg) = fsolve(self._get_f(), 
                            self.species.values,
                            fprime=fprime,
                            col_deriv=False,
                            full_output=True)

        if ier != 1:
            if use_fprime:
                return self.solve(use_fprime=False)
            else:
                #give up
                return np.array([np.NaN,]*len(self.species))

        return np.square(out)



