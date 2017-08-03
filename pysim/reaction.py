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
        def parse_reactants(stoic, spec):
            if not spec:
                return
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

        l_stoic = np.zeros(len(species))
        r_stoic = np.zeros(len(species))
        
        #find the reaction specification
        rs = cls._reaction.search(line) 
        if not rs:
            raise ReactionError("No reaction spec found")

        #parse lhs
        parse_reactants(l_stoic, line[:rs.start()])
        #parse rhs
        parse_reactants(r_stoic, line[rs.end():])

        #now parse the reaction spec
        k_rv = None
        if rs.groups()[0]:
            k_rv = params.getIndexByName(rs.groups()[0].strip())
        k_fw = params.getIndexByName(rs.groups()[1].strip())

        return Reaction(params, 
                        species,
                        l_stoic, 
                        r_stoic, 
                        k_fw, 
                        k_rv)

    def getRates(self, y):
        """Calculate the rates of change for each species due to this reaction
           given the current concentrations y"""
        #find the forward rate of the reaction
        K_fw = (self.consts.values[self.k_fw] 
                    * np.product(np.power(y,self.stoic_l)))
        #if reversable, subtract rate of reverse reaction
        if self.k_rv:
            K_fw -= (self.consts.values[self.k_rv] 
                        * np.product(np.power(y,self.stoic_r)))

        return K_fw * (self.stoic_r - self.stoic_l)

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

        spec = ""
        if self.k_rv:
            spec = "<[{}]--[{}]>".format(self.params.names[self.k_rv],
                                         self.params.names[self.k_fw])
        else:
            spec = "--[{}]>".format(self.params.names[self.k_fw])

        return " ".join([str_species(self.l_stoic),
                         spec,
                         str_species(self.r_stoic)]).strip()

