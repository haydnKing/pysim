from scipy.integrate import odeint
from scipy.optimize import fsolve, approx_fprime
from scipy.integrate import odeint
import numpy as np, re

from .reaction import Reaction
from .symbols import SymbolTable
from .exceptions import *

class ODEModel:
    def __init__(self, species, params, constraints, reactions):
        self.species = species
        self.constraints = constraints
        self.params = params
        self.reactions = reactions

    @classmethod
    def fromFile(cls, filename):
        species = SymbolTable(float, 0.0)
        constraints = []
        params = SymbolTable(float, 0.0)
        reactions = []

        kw = {'species': species,
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
                                name in params):
                                raise DuplicateNameError(name)
                            kw[syms[0]].addSymbol(name, value)

                    elif syms[0] == "constraint":
                        c = " ".join(syms[1:])
                        constraints.append((c, compile(c, '<string>', 'eval')))

                    else: #reaction
                        reactions.append(
                            Reaction.fromStr(line, 
                                             params, 
                                             species))
                except ParseError as p:
                    p.setLine(i+1)
                    raise
        return cls(species, params, constraints, reactions)

    def __str__(self):
        lines = [
            'param ' + str(self.params),
            'species ' + str(self.species),
        ]
        if self.constraints:
            for c in self.constraints:
                lines.append("constraint " + c[0])

        lines += [str(r) for r in self.reactions]
        return '\n'.join(lines) + '\n'

    def _get_unwrapped_f(self):
        #stoichiometry matrix len(species)xlen(reactions)
        S = np.array([r.getStoiciometry() for r in self.reactions]).T
        rates = [r.getRateEquation() for r in self.reactions]
        def f(x):
            ret = np.array([r(x) for r in rates])
            return S.dot(ret)[:len(self.species)]
        return f

    def _get_constraints(self):
        if not self.constraints:
            return []

        glob = {'sqrt': np.sqrt,
                'log': np.log,}
        L = len(self.constraints)

        def f(constraint):
            return lambda x: eval(constraint,
                                  glob,
                                  {k:v for k,v in zip(self.species.names, x)})

        return [f(c[1]) for c in self.constraints]

    def _get_f(self):
        f = self._get_unwrapped_f()
        C = self._get_constraints()
        eps = np.sqrt(np.finfo(float).eps)
        S = len(self.species)
        L = len(self.constraints)

        def g(y):
            z = y[:S]
            for i in range(S):
                if z[i] == 0:
                    z[i] = eps
            x = np.square(z)
            return np.append(f(x) / (z * 2),
                             [c(x) for c in C])
        return g

    def _get_unwrapped_fprime(self):
        #stoichiometry matrix len(species)xlen(reactions)
        S = np.array([r.getStoiciometry() for r in self.reactions]).T
        J = [r.getJacobianEquation() for r in self.reactions]
        def j(y):
            return S.dot(np.array([r(y) for r in J]))
                                                     
        return j
    
    def _get_fprime(self):
        j = self._get_unwrapped_fprime()
        f = self._get_f()
        C = self._get_constraints()
        S = len(self.species)
        L = len(self.constraints)
        eps = np.sqrt(np.finfo(float).eps)

        def g(y):
            z = y[:S]
            x = np.square(z)
            J = np.zeros((S+L, S+L))
            J[:S,:S] = j(x)
            F = f(z)
            for m in range(S):
                for n in range(S):
                    if n == m:
                        J[m,n] -= F[m] / z[m]
                    else:
                        J[m,n] *= z[n] / z[m]

            if L:
                J[S:,:S] = np.array([approx_fprime(x, C[i], eps) 
                                     for i in range(L)])

            return J

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
        elif name in self.species:
           return self.species.getValueByName(name)
        else:
            raise KeyError("\"{}\" is not a known species or parameter"
                           .format(name))

    def integrate(self, x_0, time=1000.0):
        f = self._get_unwrapped_f()
        J = self._get_unwrapped_fprime()

        y, info = odeint(lambda x,t: f(x),
                         x_0,
                         t = np.array([0, time]),
                         full_output=True,
                         Dfun=lambda x,t: J(x))

        return y[-1,:]

    def solve(self, use_fprime=True):
        """Calculate the steady state solution of the system of equations"""
        
        fprime = None
        if use_fprime:
            fprime = self._get_fprime()

        xi = self.integrate(self.species.values)
        initial = np.append(np.sqrt(xi), [1.0,]*len(self.constraints))

        (out, info, ier, mesg) = fsolve(self._get_f(), 
                                        initial,
                                        fprime=fprime,
                                        col_deriv=False,
                                        full_output=True)

        if ier != 1:
            if use_fprime:
                return self.solve(use_fprime=False)
            else:
                #give up
                return np.array([np.NaN,]*len(self.species))

        return np.square(out[:len(self.species)])



