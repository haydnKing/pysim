from exceptions import *
import numpy as np, re

class SymbolTable:
    def __init__(self):
        self.names = []
        self.values = np.array([])
        self._r = re.compile("[\w\^]+")

    def addSymbol(self, name, value=0):
        self._check_valid(name)
        if name in self.names:
            raise DuplicateNameError(name)
        else:
            self.names.append(name)
            self.values = np.append(self.values, value)

    def addFromStr(self, string):
        syms = [s.strip() for s in string.split('=')]
        self.addSymbol(*syms)

    def __contains__(self, name):
        return name in self.names

    def getValueByName(self, name):
        try:
            return self.values[self.names.index(name)]
        except ValueError:
            raise NameLookupError(name)

    def getIndexByName(self, name):
        try:
            return self.names.index(name)
        except ValueError:
            raise NameLookupError(name)

    def getValues(self):
        return self.values

    def getNames(self):
        return self.names

    def _check_valid(self, name):
        if not self._r.match(name):
            raise InvalidNameError(name)

    def __len__(self):
        return len(self.names)

