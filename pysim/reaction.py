import numpy as np, re
from .exceptions import *
from .symbols import SymbolTable

class Reaction:
    _reaction = re.compile(r"(?:<\[(.+)\])?--\[(.+)\]>")
    _component = re.compile(r"(\d+)?\s*([a-zA-Z_^][a-zA-Z0-9_\{\}^]*)")
    """A reaction within the model"""
    def __init__(self, 
                 params, 
                 species, 
                 l_stoic, 
                 l_const, 
                 r_stoic, 
                 r_const, 
                 k_fw, 
                 k_rv=None):
        """Define the reaction.
            const_symbols: SymbolTable to use
            l_stoic: np.array of left hand stoichiometry
            r_stoic: np.array of right hand stoichiometry
            k_fw: index of forward rate in const_table
            k_rv: index of reverse rate in const_table, None in irreversable
                reaction
        """
        self.k_fw = k_fw
        self.k_rv = k_rv
        self.params = params
        self.species = species
        self.l_stoic = l_stoic
        self.l_const = l_const
        self.r_stoic = r_stoic
        self.r_const = r_const

    @classmethod
    def fromStr(cls, line, params, species, consts):
        """Utility function to build reaction from string definition
        reaction_str = "[reactant_def]* reaction_spec [reactant_def]*"
        where:
            reactant_def := + [stoichiometry=1] species_name
            nb. the first '+' may be omitted from a list of reactant_defs
            reaction_spec = [<\[rv_rate_symbol\]]--\[fw_rate_symbol\]>
            """
        
        #find the reaction specification
        rs = cls._reaction.search(line) 
        if not rs:
            raise ReactionError("No reaction spec found")

        #parse lhs
        l_stoic, l_const = cls._parse_reactants(species, consts, line[:rs.start()])
        #parse rhs
        r_stoic, r_const = cls._parse_reactants(species, consts, line[rs.end():])

        #now parse the reaction spec
        k_rv = []
        if rs.groups()[0]:
            k_rv = [s.strip() for s in rs.groups()[0].split(',')]
        k_fw = [s.strip() for s in rs.groups()[1].split(',')]

        return Reaction(params, 
                        species,
                        l_stoic, 
                        l_const,
                        r_stoic,
                        r_const,
                        cls._parse_rateeq(k_fw, params, species, consts), 
                        cls._parse_rateeq(k_rv, params, species, consts))
    
    @classmethod
    def _parse_reactants(cls, species, consts, spec):
        """parse a string of reactants
            species: species SymbolTable 
            spec: reactant string (e.g. "3f + 5g")
        """
        stoic = np.zeros(len(species))
        c_stoic = np.zeros(len(consts))
        if not spec:
            return stoic, c_stoic
        #remove optional first '+'
        if spec[0] == '+':
            spec = spec[1:]
        for symbol in (s.strip() for s in spec.split('+')):
            if not symbol:
                raise ExpectedSymbolError("Missing reaction component")
            m = cls._component.match(symbol)
            if not m:
                raise ReactionError(
                    "Couldn't parse reaction component from \"{}\"".format(
                        symbol))
            name = m.groups()[1]
            if name in species:
                index = species.getIndexByName(name)
                if m.groups()[0]:
                    stoic[index] += int(m.groups()[0])
                else:
                    stoic[index] += 1
            elif name in consts:
                index = consts.getIndexByName(name)
                if m.groups()[0]:
                    c_stoic[index] += int(m.groups()[0])
                else:
                    c_stoic[index] += 1
            else:
                raise NameLookupError(name)

        return stoic, c_stoic

    @staticmethod
    def _parse_rateeq(args, params, species, consts):
        #zero rate
        if not args:
            return []
        #proportional rate
        if len(args) == 1:
            return [params.getIndexByName(args[0]),]
        #michaelis-menten
        elif len(args) == 3:
            if not (args[0] in species or args[0] in consts):
                raise NameLookupError(args[0])
            return [args[0],
                    params.getIndexByName(args[1]),
                    params.getIndexByName(args[2]),]
        raise ReactionError("Cannot parse reaction rate")

    @staticmethod
    def _get_rateeq(args, species, params, consts, C):
        """return a function which calculates 1directional rate"""
        if not args:
            return lambda y,s: 0
        if len(args) == 1:
            k = params.values[args[0]]
            return lambda y,s: k * (s*C)
        k_cat = params.values[args[1]]
        k_m = params.values[args[2]]
        if args[0] in species:
            e = species.getIndexByName(args[0])
            return lambda y,s: (y[e]*k_cat*(s*C))/(k_m+(s*C))
        #args[0] in consts
        e = consts.getValueByName(args[0])
        return lambda y,s: (e*k_cat*(s*C))/(k_m+(s*C))

    @staticmethod
    def _get_jacobian(args, species, consts, params, stoic):
        """return function to calculate row of jacobian from this reaction"""
        if not args:
            return lambda x: np.zeros(len(stoic))
        if len(args) == 1:
            k = params.values[args[0]]
            I = range(len(stoic))
            def j(X): 
                a = np.array([
                    stoic[i] * k * np.power(X[i],stoic[i]-1) *
                    np.product(np.power(np.delete(X,i), np.delete(stoic,i)))
                    if stoic[i] else 0.0 for i in I])
                return a
            return j

        idx=-1
        if args[0] in consts:
            val = consts.getValueByName(args[0])
            e = lambda X: val
        else:
            idx = species.getIndexByName(args[0])
            e = lambda X: X[idx]
        k_cat = params.values[args[1]]
        k_m = params.values[args[2]]
        I = range(len(stoic))
        def j(X):
            s = np.product(np.power(X, stoic))
            a = 1./(k_m + s)
            b = (e(X) * k_cat * s) * a * a
            dh = np.array([stoic[i] * np.power(X[i],stoic[i]-1) * 
                           np.product(np.power(np.delete(X,i), 
                                               np.delete(stoic,i)))
                           if stoic[i] else 0.0 for i in I])
            dg = e(X) * k_cat*dh
            #special case
            if idx >= 0:
                dg[idx] = (stoic[idx]+1)*k_cat*np.product(np.power(X,stoic))
            return a*dg - b*dh
        return j

    def getStoiciometry(self):
        return self.r_stoic - self.l_stoic

    def getRateEquation(self, consts):
        """Return a function which calculates the rates of change for each 
            species due to this reaction given the current concentrations y"""
        l_c = np.product(np.power(consts.values,self.l_const))
        r_c = np.product(np.power(consts.values,self.r_const))
        f_fw = self._get_rateeq(self.k_fw, self.species, self.params, consts, l_c)
        f_rv = self._get_rateeq(self.k_rv, self.species, self.params, consts, r_c)

        def fn(x):
            #find the forward rate of the reaction
            K_fw = f_fw(x, np.product(np.power(x,self.l_stoic)))
            #if reversable, subtract rate of reverse reaction
            K_fw -= f_rv(x, np.product(np.power(x,self.r_stoic)))

            return K_fw

        return fn

    def getJacobianEquation(self, species, consts):
        """Return a function which calculates the rates of change for each 
            species due to this reaction given the current concentrations y"""
        f_fw = self._get_jacobian(self.k_fw, species, consts, self.params, self.l_stoic)
        f_rv = self._get_jacobian(self.k_rv, species, consts, self.params, self.r_stoic)

        def fn(x):
            #find the forward part of the jacobian
            J = f_fw(x)
            #if reversable, subtract the reverse part of the jacobian
            J -= f_rv(x)

            return J 

        return fn

    def __str__(self):
        def str_species(stoic):
            out = []
            for i,s in enumerate(stoic):
                if s != 0:
                    out.append('+')
                    if s != 1:
                        if s == int(s):
                            out.append(str(int(s)))
                        else:
                            out.append(str(s))
                    out.append(self.species.names[i])
            return " ".join(out[1:])

        def str_rate(r):
            if not r:
                return ""
            if len(r) == 1:
                return self.params.names[r[0]]
            return ", ".join([self.species.names[r[0]],
                              self.params.names[r[1]],
                              self.params.names[r[2]]])

        spec = ""
        if self.k_rv:
            spec = "<[{}]--[{}]>".format(str_rate(self.k_rv),
                                         str_rate(self.k_fw))
        else:
            spec = "--[{}]>".format(str_rate(self.k_fw))

        return " ".join([str_species(self.l_stoic),
                         spec,
                         str_species(self.r_stoic)]).strip()

