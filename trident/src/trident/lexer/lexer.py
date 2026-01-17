"""
Lexer implementation for the Trident programming language.

The lexer converts source code into a stream of tokens, handling:
- Significant whitespace (Python-style indentation)
- Multi-line strings for natural language blocks
- All Trident operators and delimiters
- Comments and whitespace
"""

from dataclasses import dataclass, field
from typing import Iterator, Generator

from trident.lexer.tokens import (
    Token,
    TokenType,
    SourceLocation,
    KEYWORDS,
    ANNOTATION_KEYWORDS,
)


class LexerError(Exception):
    """Exception raised for lexer errors."""
    
    def __init__(self, message: str, location: SourceLocation) -> None:
        self.message = message
        self.location = location
        super().__init__(f"{location}: {message}")


@dataclass
class Lexer:
    """
    Lexer for the Trident programming language.
    
    Converts source code into a stream of tokens with proper handling
    of Python-style significant indentation.
    
    Usage:
        lexer = Lexer(source_code)
        tokens = list(lexer.tokenize())
    """
    
    source: str
    filename: str = "<stdin>"
    
    # Internal state
    _pos: int = field(default=0, init=False, repr=False)
    _line: int = field(default=1, init=False, repr=False)
    _column: int = field(default=1, init=False, repr=False)
    _indent_stack: list[int] = field(default_factory=lambda: [0], init=False, repr=False)
    _at_line_start: bool = field(default=True, init=False, repr=False)
    _pending_tokens: list[Token] = field(default_factory=list, init=False, repr=False)
    _nesting_level: int = field(default=0, init=False, repr=False)
    
    @property
    def _current(self) -> str:
        """Get current character or empty string if at end."""
        if self._pos >= len(self.source):
            return ""
        return self.source[self._pos]
    
    @property
    def _peek(self) -> str:
        """Peek at next character."""
        if self._pos + 1 >= len(self.source):
            return ""
        return self.source[self._pos + 1]
    
    def _location(self) -> SourceLocation:
        """Get current source location."""
        return SourceLocation(self._line, self._column, self.filename)
    
    def _advance(self) -> str:
        """Advance position and return current character."""
        char = self._current
        self._pos += 1
        if char == "\n":
            self._line += 1
            self._column = 1
        else:
            self._column += 1
        return char
    
    def _make_token(self, token_type: TokenType, value: any, raw: str) -> Token:
        """Create a token at the current position."""
        return Token(
            type=token_type,
            value=value,
            location=self._location(),
            raw=raw,
        )
    
    def _skip_whitespace_inline(self) -> None:
        """Skip inline whitespace (spaces and tabs, not newlines)."""
        while self._current in (" ", "\t"):
            self._advance()
    
    def _skip_comment(self) -> None:
        """Skip a comment until end of line."""
        while self._current and self._current != "\n":
            self._advance()
    
    def _read_string(self, quote: str) -> Token:
        """Read a string literal."""
        location = self._location()
        start_pos = self._pos
        
        # Check for triple-quoted string
        if self._current == quote and self._peek == quote:
            self._advance()  # second quote
            self._advance()  # third quote
            return self._read_triple_string(quote, location, start_pos)
        
        # Regular string
        chars: list[str] = []
        self._advance()  # opening quote
        
        while self._current and self._current != quote:
            if self._current == "\n":
                raise LexerError("Unterminated string literal", location)
            if self._current == "\\":
                self._advance()
                chars.append(self._read_escape_sequence())
            else:
                chars.append(self._advance())
        
        if not self._current:
            raise LexerError("Unterminated string literal", location)
        
        self._advance()  # closing quote
        value = "".join(chars)
        raw = self.source[start_pos:self._pos]
        
        return Token(TokenType.STRING, value, location, raw)
    
    def _read_triple_string(self, quote: str, location: SourceLocation, start_pos: int) -> Token:
        """Read a triple-quoted string (multi-line)."""
        self._advance()  # after third opening quote
        chars: list[str] = []
        
        while self._current:
            if (self._current == quote and 
                self._peek == quote and 
                self._pos + 2 < len(self.source) and 
                self.source[self._pos + 2] == quote):
                # Found closing triple quote
                self._advance()
                self._advance()
                self._advance()
                break
            
            if self._current == "\\":
                self._advance()
                chars.append(self._read_escape_sequence())
            else:
                chars.append(self._advance())
        else:
            raise LexerError("Unterminated triple-quoted string", location)
        
        value = "".join(chars)
        raw = self.source[start_pos:self._pos]
        
        return Token(TokenType.TRIPLE_STRING, value, location, raw)
    
    def _read_escape_sequence(self) -> str:
        """Read an escape sequence and return the escaped character."""
        escape_chars = {
            "n": "\n",
            "t": "\t",
            "r": "\r",
            "\\": "\\",
            "'": "'",
            '"': '"',
            "0": "\0",
        }
        
        char = self._advance()
        if char in escape_chars:
            return escape_chars[char]
        elif char == "x":
            # Hex escape \xNN
            hex_chars = self._advance() + self._advance()
            try:
                return chr(int(hex_chars, 16))
            except ValueError:
                raise LexerError(f"Invalid hex escape: \\x{hex_chars}", self._location())
        elif char == "u":
            # Unicode escape \uNNNN
            hex_chars = "".join(self._advance() for _ in range(4))
            try:
                return chr(int(hex_chars, 16))
            except ValueError:
                raise LexerError(f"Invalid unicode escape: \\u{hex_chars}", self._location())
        else:
            # Unknown escape, keep as-is
            return char
    
    def _read_number(self) -> Token:
        """Read a numeric literal (integer or float)."""
        location = self._location()
        start_pos = self._pos
        chars: list[str] = []
        is_float = False
        
        # Handle negative sign
        if self._current == "-":
            chars.append(self._advance())
        
        # Integer part
        while self._current.isdigit():
            chars.append(self._advance())
        
        # Decimal part
        if self._current == "." and self._peek.isdigit():
            is_float = True
            chars.append(self._advance())  # .
            while self._current.isdigit():
                chars.append(self._advance())
        
        # Exponent part
        if self._current in ("e", "E"):
            is_float = True
            chars.append(self._advance())
            if self._current in ("+", "-"):
                chars.append(self._advance())
            while self._current.isdigit():
                chars.append(self._advance())
        
        raw = "".join(chars)
        
        if is_float:
            value = float(raw)
            return Token(TokenType.FLOAT, value, location, raw)
        else:
            value = int(raw)
            return Token(TokenType.INTEGER, value, location, raw)
    
    def _read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        location = self._location()
        chars: list[str] = []
        
        while self._current.isalnum() or self._current == "_":
            chars.append(self._advance())
        
        raw = "".join(chars)
        
        # Check if it's a keyword
        token_type = KEYWORDS.get(raw, TokenType.IDENTIFIER)
        
        return Token(token_type, raw, location, raw)
    
    def _read_annotation(self) -> Token:
        """Read an annotation after @."""
        location = self._location()
        self._advance()  # skip @
        
        # Read the annotation name
        chars: list[str] = []
        while self._current.isalnum() or self._current == "_":
            chars.append(self._advance())
        
        name = "".join(chars)
        
        # Check if it's a known annotation keyword
        if name in ANNOTATION_KEYWORDS:
            return Token(ANNOTATION_KEYWORDS[name], name, location, f"@{name}")
        
        # Otherwise return as AT token followed by identifier
        # We need to handle this specially - put identifier back in pending
        self._pending_tokens.append(Token(TokenType.IDENTIFIER, name, location, name))
        return Token(TokenType.AT, "@", location, "@")
    
    def _handle_indentation(self) -> list[Token]:
        """Handle indentation at start of line, producing INDENT/DEDENT tokens."""
        tokens: list[Token] = []
        location = self._location()
        
        # Count indentation
        indent = 0
        while self._current in (" ", "\t"):
            if self._current == " ":
                indent += 1
            else:  # tab
                indent += 4  # tabs = 4 spaces
            self._advance()
        
        # Skip blank lines and comments
        if self._current in ("\n", "#", ""):
            return tokens
        
        current_indent = self._indent_stack[-1]
        
        if indent > current_indent:
            self._indent_stack.append(indent)
            tokens.append(Token(TokenType.INDENT, indent, location, ""))
        elif indent < current_indent:
            while self._indent_stack and self._indent_stack[-1] > indent:
                self._indent_stack.pop()
                tokens.append(Token(TokenType.DEDENT, indent, location, ""))
            if self._indent_stack[-1] != indent:
                raise LexerError(f"Inconsistent indentation: {indent}", location)
        
        return tokens
    
    def _read_operator_or_delimiter(self) -> Token:
        """Read operators and delimiters."""
        location = self._location()
        char = self._current
        
        # Two-character operators
        two_char = char + self._peek
        two_char_ops = {
            "==": TokenType.EQ,
            "!=": TokenType.NE,
            "<=": TokenType.LE,
            ">=": TokenType.GE,
            "->": TokenType.ARROW,
            "//": TokenType.DOUBLE_SLASH,
            "**": TokenType.DOUBLE_STAR,
            "+=": TokenType.PLUS_ASSIGN,
            "-=": TokenType.MINUS_ASSIGN,
            "*=": TokenType.STAR_ASSIGN,
            "/=": TokenType.SLASH_ASSIGN,
        }
        
        if two_char in two_char_ops:
            self._advance()
            self._advance()
            return Token(two_char_ops[two_char], two_char, location, two_char)
        
        # Single-character operators and delimiters
        single_char_ops = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
            "<": TokenType.LT,
            ">": TokenType.GT,
            "=": TokenType.ASSIGN,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            ",": TokenType.COMMA,
            ":": TokenType.COLON,
            ";": TokenType.SEMICOLON,
            ".": TokenType.DOT,
        }
        
        if char in single_char_ops:
            self._advance()
            
            # Update nesting level for implicit line joining
            if char in "([{":
                self._nesting_level += 1
            elif char in ")]}":
                self._nesting_level = max(0, self._nesting_level - 1)
                
            return Token(single_char_ops[char], char, location, char)
        
        raise LexerError(f"Unexpected character: {char!r}", location)
    
    def tokenize(self) -> Generator[Token, None, None]:
        """
        Tokenize the source code.
        
        Yields tokens one at a time, handling indentation and producing
        INDENT/DEDENT tokens as needed.
        """
        while self._pos < len(self.source) or self._pending_tokens:
            # Return any pending tokens first
            while self._pending_tokens:
                yield self._pending_tokens.pop(0)
            
            if self._pos >= len(self.source):
                break
            
            # Handle start of line (indentation)
            if self._at_line_start:
                self._at_line_start = False
                # Only check indentation if not inside brackets (implicit line joining)
                if self._nesting_level == 0:
                    indent_tokens = self._handle_indentation()
                    for token in indent_tokens:
                        yield token
                else:
                    # Inside brackets, just consume leading whitespace
                    while self._current in (" ", "\t"):
                        self._advance()
            
            # Skip inline whitespace
            self._skip_whitespace_inline()
            
            if self._pos >= len(self.source):
                break
            
            char = self._current
            
            # Comments
            if char == "#":
                self._skip_comment()
                continue
            
            # Newlines
            if char == "\n":
                yield self._make_token(TokenType.NEWLINE, "\n", "\n")
                self._advance()
                self._at_line_start = True
                continue
            
            # Strings
            if char in ('"', "'"):
                yield self._read_string(char)
                continue
            
            # Numbers
            if char.isdigit() or (char == "-" and self._peek.isdigit()):
                yield self._read_number()
                continue
            
            # Identifiers and keywords
            if char.isalpha() or char == "_":
                yield self._read_identifier()
                continue
            
            # Annotations
            if char == "@":
                # Check if it's a decorator or matmul operator
                # If at line start or after newline, it's a decorator
                yield self._read_annotation()
                continue
            
            # Operators and delimiters
            yield self._read_operator_or_delimiter()
        
        # Emit remaining DEDENT tokens at end of file
        location = self._location()
        while len(self._indent_stack) > 1:
            self._indent_stack.pop()
            yield Token(TokenType.DEDENT, 0, location, "")
        
        # EOF token
        yield Token(TokenType.EOF, None, location, "")
    
    def __iter__(self) -> Iterator[Token]:
        """Allow iteration over tokens."""
        return self.tokenize()


def lex(source: str, filename: str = "<stdin>") -> list[Token]:
    """
    Convenience function to lex source code into a list of tokens.
    
    Args:
        source: The source code to tokenize
        filename: Optional filename for error reporting
    
    Returns:
        List of all tokens
    """
    lexer = Lexer(source, filename)
    return list(lexer.tokenize())
