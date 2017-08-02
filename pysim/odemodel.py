from scipy.integrate import odeint
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
                            species.addFromStr(declaration)
                    elif syms[0] == "param":
                        for declaration in " ".join(syms[1:]).split(','):
                            params.addFromStr(declaration)
                    else: #reaction
                        reactions.append(Reaction.fromStr(line, species, params))
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

#    def _get_system_fn(self, params):
#        self._curr_params = params
#        reactions = self.define_reactions()
#        def f(y, t):
#            ret = np.zeros(len(y))
#            for r in reactions:
#                ret += r.getRates(y)
#                return ret
#        return f
#
#    def _simulate(self, 
#                  end_time, 
#                  timestep = 0.1,
#                  initial_values = {},
#                  params = {}):
#        #copy species
#        sim_vars = copy.copy(self.initial_values)
#        for k,v in initial_values.items():
#            #check for duff initial_values
#            if k not in self.species_names:
#                raise ValueError('Unknown species \'{}\''.format(k))
#            sim_vars[self.species_names.index(k)] = v 
#
#        #copy parameters
#        sim_params = copy.copy(self.parameters)
#        for k,v in params.items():
#            #check for unknown parameters
#            if k not in self.parameters:
#                raise ValueError('Unknown parameter \'{}\''.format(k))
#            sim_params[k] = v
#
#        t_out = np.arange(0, end_time, timestep)
#        y_out = odeint(self._get_system_fn(sim_params),
#                       sim_vars,
#                       t_out)
#
#        return pd.DataFrame(data=y_out,
#                            index=t_out,
#                            columns=self.species_names)
#
#    def simulate(self, 
#                 end_time, 
#                 timestep = 0.1,
#                 initial_values = {},
#                 params = {}):
#        self.save_data(initial_values,
#                       params,
#                       self._simulate(end_time, 
#                                      timestep,
#                                      initial_values,
#                                      params))

