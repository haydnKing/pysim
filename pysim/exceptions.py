

class ParseError(Exception):
    def __init__(self, message):
        super(ParseError, self).__init__(message)
        self.line = -1

    def setLine(self, line):
        self.line = line

    def append(self, appended):
        self.args[0] = self.args[0] + " " + appended

    def __str__(self):
        if self.line < 0:
            return self.args[0]
        return "line {}: {}".format(self.line, self.args[0])

class NameLookupError(ParseError):
    def __init__(self, name):
        super(NameLookupError, self).__init__(
            "Unknown name \"{}\"".format(name))

class DuplicateNameError(ParseError):
    def __init__(self, name):
        super(NameLookupError, self).__init__(
            "Redefinition of name \"{}\"".format(name))

class InvalidNameError(ParseError):
    def __init__(self, name):
        super(InvalidNameError, self).__init__(
            "Invalid name \"{}\"".format(name))

class UnexpectedSymbolError(ParseError):
    def __init__(self, name):
        super(UnexpectedSymbolError, self).__init__(
            "Unexpected symbol \"{}\"".format(name))

class ExpectedSymbolError(ParseError):
    def __init__(self, symbol):
        super(ExpectedSymbolError, self).__init__(
            "Expected {}".format(symbol))

class ReactionError(ParseError):
    def __init__(self, msg):
        super(ReactionError, self).__init__(
            "Couldn't parse reaction, {}".format(msg))

class SyntaxParseError(ParseError):
    def __init__(self, item, value):
        super(SyntaxParseError, self).__init__(
            "Couldn't parse {} \"{}\"".format(item, value))



