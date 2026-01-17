"""
Token definitions for the Trident programming language.

This module defines all token types and the Token class used by the lexer.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    """All token types in the Trident language."""
    
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    TRIPLE_STRING = auto()  # Multi-line strings for natural language
    BOOLEAN = auto()
    
    # Identifiers and Keywords
    IDENTIFIER = auto()
    
    # Keywords - Control Flow
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    FOR = auto()
    WHILE = auto()
    IN = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    PASS = auto()
    
    # Keywords - Definitions
    FN = auto()
    PIPELINE = auto()
    CLASS = auto()
    LET = auto()
    CONST = auto()
    STRUCT = auto()
    
    # Keywords - Imports
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    
    # Keywords - Special
    AND = auto()
    OR = auto()
    NOT = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()
    
    # Type Keywords
    TENSOR = auto()
    STRING_TYPE = auto()
    INT_TYPE = auto()
    FLOAT_TYPE = auto()
    BOOL_TYPE = auto()
    IMAGE = auto()
    DOCUMENT = auto()
    
    # Annotations (decorators)
    AT = auto()  # @
    INTENT = auto()
    MODEL = auto()
    HARDWARE = auto()
    NATURAL_LANGUAGE = auto()
    
    # Operators - Arithmetic
    PLUS = auto()       # +
    MINUS = auto()      # -
    STAR = auto()       # *
    SLASH = auto()      # /
    DOUBLE_SLASH = auto()  # //
    PERCENT = auto()    # %
    DOUBLE_STAR = auto()   # **
    MATMUL = auto()     # @ (context-dependent, also used for decorators)
    
    # Operators - Comparison
    EQ = auto()         # ==
    NE = auto()         # !=
    LT = auto()         # <
    GT = auto()         # >
    LE = auto()         # <=
    GE = auto()         # >=
    
    # Operators - Assignment
    ASSIGN = auto()     # =
    PLUS_ASSIGN = auto()    # +=
    MINUS_ASSIGN = auto()   # -=
    STAR_ASSIGN = auto()    # *=
    SLASH_ASSIGN = auto()   # /=
    
    # Delimiters
    LPAREN = auto()     # (
    RPAREN = auto()     # )
    LBRACKET = auto()   # [
    RBRACKET = auto()   # ]
    LBRACE = auto()     # {
    RBRACE = auto()     # }
    COMMA = auto()      # ,
    COLON = auto()      # :
    SEMICOLON = auto()  # ;
    DOT = auto()        # .
    ARROW = auto()      # ->
    
    # Whitespace and Structure
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    
    # Special
    EOF = auto()
    COMMENT = auto()
    ERROR = auto()


# Keywords mapping
KEYWORDS: dict[str, TokenType] = {
    # Control flow
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "elif": TokenType.ELIF,
    "for": TokenType.FOR,
    "while": TokenType.WHILE,
    "in": TokenType.IN,
    "return": TokenType.RETURN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "pass": TokenType.PASS,
    
    # Definitions
    "fn": TokenType.FN,
    "pipeline": TokenType.PIPELINE,
    "class": TokenType.CLASS,
    "let": TokenType.LET,
    "let": TokenType.LET,
    "const": TokenType.CONST,
    "struct": TokenType.STRUCT,
    
    # Imports
    "import": TokenType.IMPORT,
    "from": TokenType.FROM,
    "as": TokenType.AS,
    
    # Boolean and None
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "true": TokenType.TRUE,
    "True": TokenType.TRUE,
    "false": TokenType.FALSE,
    "False": TokenType.FALSE,
    "none": TokenType.NONE,
    "None": TokenType.NONE,
    
    # Types
    "Tensor": TokenType.TENSOR,
    "String": TokenType.STRING_TYPE,
    "Int": TokenType.INT_TYPE,
    "Float": TokenType.FLOAT_TYPE,
    "Bool": TokenType.BOOL_TYPE,
    "Image": TokenType.IMAGE,
    "Image": TokenType.IMAGE,
    "Document": TokenType.DOCUMENT,

    # Lowercase aliases for better DX
    "tensor": TokenType.TENSOR,
    "string": TokenType.STRING_TYPE,
    "int": TokenType.INT_TYPE,
    "float": TokenType.FLOAT_TYPE,
    "bool": TokenType.BOOL_TYPE,
    "image": TokenType.IMAGE,
    "document": TokenType.DOCUMENT,
}

# Annotation keywords (after @)
ANNOTATION_KEYWORDS: dict[str, TokenType] = {
    "intent": TokenType.INTENT,
    "model": TokenType.MODEL,
    "hardware": TokenType.HARDWARE,
    "natural_language": TokenType.NATURAL_LANGUAGE,
}


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Location in source code for error reporting."""
    line: int
    column: int
    file: str = "<stdin>"
    
    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass(frozen=True, slots=True)
class Token:
    """
    A token produced by the lexer.
    
    Attributes:
        type: The type of token
        value: The literal value (for literals) or lexeme
        location: Source location for error reporting
        raw: The raw string from source code
    """
    type: TokenType
    value: Any
    location: SourceLocation
    raw: str
    
    def __str__(self) -> str:
        if self.type in (TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT):
            return f"Token({self.type.name})"
        return f"Token({self.type.name}, {self.value!r})"
    
    def __repr__(self) -> str:
        return f"Token(type={self.type.name}, value={self.value!r}, location={self.location})"
    
    def is_literal(self) -> bool:
        """Check if token is a literal value."""
        return self.type in (
            TokenType.INTEGER,
            TokenType.FLOAT,
            TokenType.STRING,
            TokenType.TRIPLE_STRING,
            TokenType.TRUE,
            TokenType.FALSE,
            TokenType.NONE,
        )
    
    def is_operator(self) -> bool:
        """Check if token is an operator."""
        return self.type in (
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.DOUBLE_SLASH, TokenType.PERCENT, TokenType.DOUBLE_STAR,
            TokenType.MATMUL, TokenType.EQ, TokenType.NE, TokenType.LT,
            TokenType.GT, TokenType.LE, TokenType.GE, TokenType.AND,
            TokenType.OR, TokenType.NOT,
        )
    
    def is_keyword(self) -> bool:
        """Check if token is a keyword."""
        return self.type in KEYWORDS.values()
