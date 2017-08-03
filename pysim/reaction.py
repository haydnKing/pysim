import numpy as np, re
from .exceptions import *
from .symbols import SymbolTable

class Reaction:
    _reaction = re.compile(r"(?:<\[(.+)\])?--\[(.+)\]>")
    _component = re.compile(r"(\d+)?\s*([a-zA-Z_^][a-zA-Z0-9_\{\}^]*)")
    """A reaction within the model"""
    def __init__(self, params, species, l_stoic, r_stoic, k_fw, k_rv=None):
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
        self.r_stoic = r_stoic

    @classmethod
    def fromStr(cls, line, params, species):
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
        l_stoic = cls._parse_reactants(species, line[:rs.start()])
        #parse rhs
        r_stoic = cls._parse_reactants(species, line[rs.end():])

        #now parse the reaction spec
        k_rv = []
        if rs.groups()[0]:
            k_rv = [s.strip() for s in rs.groups()[0].split(',')]
        k_fw = [s.strip() for s in rs.groups()[1].split(',')]

        return Reaction(params, 
                        species,
                        l_stoic, 
                        r_stoic, 
                        cls._parse_rateeq(k_fw, params, species), 
                        cls._parse_rateeq(k_rv, params, species))
    
    @classmethod
    def _parse_reactants(cls, species, spec):
        """parse a string of reactants
            species: species SymbolTable 
            spec: reactant string (e.g. "3f + 5g")
        """
        stoic = np.zeros(len(species))
        if not spec:
            return stoic
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
            index = species.getIndexByName(m.groups()[1])
            if m.groups()[0]:
                stoic[index] += int(m.groups()[0])
            else:
                stoic[index] += 1
        return stoic

    @staticmethod
    def _parse_rateeq(args, params, species):
        #zero rate
        if not args:
            return []
        #proportional rate
        if len(args) == 1:
            return [params.getIndexByName(args[0]),]
        #michaelis-menten
        elif len(args) == 3:
            return [species.getIndexByName(args[0]),
                    params.getIndexByName(args[1]),
                    params.getIndexByName(args[2]),]
        raise ReactionError("Cannot parse reaction rate")

    @staticmethod
    def _get_rateeq(args, params):
        """return a function which calculates 1directional rate"""
        if not args:
            return lambda y,s: 0
        if len(args) == 1:
            k = params.values[args[0]]
            return lambda y,s: k * s
        e = args[0]
        k_cat = params.values[args[1]]
        k_m = params.values[args[2]]
        return lambda y,s: (y[e]*k_cat*s)/(k_m+s)

    @staticmethod
    def _get_jacobian(args, params, stoic):
        """return function to calculate row of jacobian from this reaction"""
        if not args:
            return lambda x: np.zeros(len(x))
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

        e = args[0]
        k_cat = params.values[args[1]]
        k_m = params.values[args[2]]
        I = range(len(stoic))
        def j(X):
            s = np.product(np.power(X, stoic))
            a = 1./(k_m + s)
            b = (X[e] * k_cat * s) * a * a
            dh = np.array([stoic[i] * np.power(X[i],stoic[i]-1) * 
                           np.product(np.power(np.delete(X,i), 
                                               np.delete(stoic,i)))
                           for i in I])
            dg = X[e] * k_cat*dh
            #special case
            dg[e] = (stoic[e]+1)*k_cat*np.product(np.power(X,stoic))
            print("j({}) = {}".format(X, a*dg - b*dh))
            return a*dg - b*dh
        return j



    def getStoiciometry(self):
        return self.r_stoic - self.l_stoic

    def getRateEquation(self):
        """Return a function which calculates the rates of change for each 
            species due to this reaction given the current concentrations y"""
        f_fw = self._get_rateeq(self.k_fw, self.params)
        f_rv = self._get_rateeq(self.k_rv, self.params)

        def fn(species):
            #find the forward rate of the reaction
            K_fw = f_fw(species, np.product(np.power(species,self.l_stoic)))
            #if reversable, subtract rate of reverse reaction
            K_fw -= f_rv(species, np.product(np.power(species,self.r_stoic)))

            return K_fw

        return fn

    def getJacobianEquation(self):
        """Return a function which calculates the rates of change for each 
            species due to this reaction given the current concentrations y"""
        f_fw = self._get_jacobian(self.k_fw, self.params, self.l_stoic)
        f_rv = self._get_jacobian(self.k_rv, self.params, self.r_stoic)

        def fn(species):
            #find the forward part of the jacobian
            J = f_fw(species)
            #if reversable, subtract the reverse part of the jacobian
            J -= f_rv(species)

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

