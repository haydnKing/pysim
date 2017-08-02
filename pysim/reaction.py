import numpy as np, re
from .exceptions import *
from .symbols import SymbolTable

class Reaction:
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
    def fromStr(cls, line, var_symbols, const_symbols):
        """Utility function to build reaction from string definition
        reaction_str = "[reactant_def]* reaction_spec [reactant_def]*"
        where:
            reactant_def := + [stoichiometry=1] species_name
            nb. the first '+' may be omitted from a list of reactant_defs
            reaction_spec = --[fw_rate_symbol]> | <[rv_rate_symbol]--[fw_rate_symbol]>
            """
        def parse_reactants(stoic, symbols):
            if not symbols:
                return
            #add a leading + if there isn't one
            if symbols[0][0] != '+':
                symbols = ['+'] + symbols
            need_plus = True
            seen_stoic = False
            stoic_value = 1
            for symbol in symbols:
                if need_plus:
                    if symbol[0] != '+':
                        raise UnexpectedSymbolError(symbol)
                    need_plus = False
                    #if there was a + on its own
                    if len(symbol) == 1:
                        continue
                    #if the plus was attached to the symbol
                    else:
                        symbol = symbol[1:]
                if not seen_stoic:
                    #try and extract stoichiometry
                    try:
                        stoic_value = int(symbol)
                        seen_stoic = True
                        #success, move to next symbol
                        continue
                    except ValueError:
                        #no stoichiometry, must be a species
                        pass
                #parse the species
                stoic[var_symbols.getIndexByName(symbol)] += stoic_value
                #reset state
                need_plus = True
                seen_stoic = False
                stoic_value = 1
            #check iteration didn't end while we were expecting something
            if not need_plus:
                if seen_stoic:
                    raise ExpectedSymbolError("species identifier")
                raise ExpectedSymbolError("species identifier or stoichiometry")
        
        l_stoic = np.zeros(len(var_symbols))
        r_stoic = np.zeros(len(var_symbols))
        
        symbols = line.split()
        #find the reaction specification
        reaction_spec = -1
        for i,s in enumerate(symbols):
            if "--" in s:
                reaction_spec = i
                break

        #parse lhs
        parse_reactants(l_stoic, symbols[:reaction_spec])
        #parse rhs
        parse_reactants(r_stoic, symbols[reaction_spec+1:])

        #now parse the reaction spec
        m = re.match("(?:<\[([\w\^]+)\])?--\[([\w\^]+)\]>", 
                     symbols[reaction_spec])
        if not m:
            raise ReactionError("malformed reaction spec")
        k_rv = None
        if m.groups()[0]:
            k_rv = const_symbols.getIndexByName(m.groups()[0])
        k_fw = const_symbols.getIndexByName(m.groups()[1])

        return Reaction(const_symbols, 
                        var_symbols,
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

