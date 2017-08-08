from .exceptions import *
import re

class SymbolTable:
    def __init__(self):
        self.names = []
        self.values = []
        self._r = re.compile("[^\s\[\]]+")

    def copy(self):
        r = SymbolTable()
        r.names = self.names.copy()
        r.values = self.values.copy()
        return r

    def merge(self, rhs):
        r = SymbolTable()
        r.names = self.names + rhs.names
        r.values = self.values + rhs.values
        return r

    def addSymbol(self, name, value=0.):
        self._check_valid(name)
        if name in self.names:
            raise DuplicateNameError(name)
        else:
            self.names.append(name)
            self.values.append(value)

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

    def setValueByName(self, name, value):
        try:
            self.values[self.names.index(name)] = value
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

    def __str__(self):
        o = []
        for i,name in enumerate(self.names):
            if self.values[i] == 0.0:
                o.append(name)
            else:
                o.append("{}={}".format(name, self.values[i]))
        return ", ".join(o)

