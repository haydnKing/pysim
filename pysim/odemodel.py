from scipy.integrate import ode
from scipy.optimize import fsolve, approx_fprime
from scipy.integrate import odeint
import numpy as np, re

from .reaction import Reaction
from .symbols import SymbolTable
from .exceptions import *

class ODEModel:
    def __init__(self, species, params, consts, constraints, reactions):
        self.species = species
        self.constraints = constraints
        self.params = params
        self.consts = consts
        self.reactions = reactions

    @classmethod
    def fromFile(cls, filename):
        species = SymbolTable(float, 0.0)
        consts = SymbolTable(float, 0.0)
        params = SymbolTable(float, 0.0)
        constraints = []
        reactions = []

        kw = {'species': species,
              'param': params,
              'params': params,
              'const': consts,
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
                                             species,
                                             consts))
                except ParseError as p:
                    p.setLine(i+1)
                    raise
        return cls(species, params, consts, constraints, reactions)

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
        rates = [r.getRateEquation(self.consts) for r in self.reactions]

        def f(x):
            return S.dot(np.array([r(x) for r in rates]))[:len(self.species)]
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

    def set(self, k, v):
        """set params or species initial conditions"""
        if k in self.params:
           self.params.setValueByName(k,v)
        elif k in self.consts:
            self.consts.setValueByName(k,v)
        elif k in self.species:
            self.species.setValueByName(k,v)
        else:
            raise KeyError("\"{}\" is not a known species or parameter"
                           .format(k))

    def setAllParams(self, params):
        """Set all parameters"""
        if len(self.params) + len(self.consts) != len(params):
            raise ValueError("Expected {} parameters, got {}".format(
                len(self.params)+len(self.consts), len(params)))
        self.params.values = np.array(params[:len(self.params)])
        self.consts.values = np.array(params[len(self.params):])

    def setAllSpecies(self, species):
        """Set all initial conditions"""
        if len(self.species) != len(species):
            raise ValueError("Expected {} species, got {}".format(
                len(self.species), len(species)))
        self.species.values = np.array(species)

    def solveForParams(self, params, initial_conditions=None, use_fprime=True):
        self.setAllParams(params)
        return self.solve(initial_conditions, use_fprime)

    def get(self, name):
        """get params or species initial conditions"""
        if name in self.params:
           return self.params.getValueByName(name)
        elif name in self.species:
           return self.species.getValueByName(name)
        else:
            raise KeyError("\"{}\" is not a known species or parameter"
                           .format(name))

    def integrate(self, x_0=None, tol=1e-6, max_time=10.0):
        if x_0 is None:
            x_0 = np.array(self.species.values)

        f = self._get_unwrapped_f()
        J = self._get_unwrapped_fprime()

        integrator = ode(lambda t,x: f(x))#, lambda t,x:J(x)) 
        integrator.set_integrator('vode', 
                                  method='bdf',
                                  nsteps=1e4*(
                                      len(self.species) + 
                                      len(self.constraints))
                                 )
        integrator.set_initial_value(x_0, 0.0)

        dt = 1.0
        x_last = x_0
        #print("x_0 = {}".format(x_0))
        while integrator.successful() and integrator.t + dt < max_time:
            if np.linalg.norm(f(x_last)) < tol:
                break
            x_last = integrator.integrate(integrator.t+dt)

        return np.abs(x_last)

    def solve(self, initial=None, use_fprime=True, attempts = 6):
        """Calculate the steady state solution of the system of equations"""
        
        fprime = None
        if use_fprime:
            fprime = self._get_fprime()

        if initial is None:
            initial = self.species.values
        initial = np.append(np.sqrt(initial),
                            [1.0,] * len(self.constraints))


        for attempt in range(attempts):

            (out, info, ier, mesg) = fsolve(self._get_f(), 
                                            initial,
                                            fprime=fprime,
                                            col_deriv=False,
                                            full_output=True)

            if ier == 1:
                return np.square(out[:len(self.species)]), True
                
            xi = self.integrate(np.square(initial[:len(self.species)]), 
                                max_time=10.0,
                                tol=1e-6)
            initial = np.append(np.sqrt(xi), [1.0,]*len(self.constraints))

        return np.square(initial[:len(self.species)]), False



