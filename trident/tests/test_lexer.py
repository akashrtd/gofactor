"""
Tests for the Trident lexer.
"""

import pytest
from trident.lexer import Lexer, Token, TokenType


class TestLexer:
    """Test the lexer tokenization."""
    
    def test_empty_source(self):
        """Empty source produces only EOF."""
        lexer = Lexer("")
        tokens = list(lexer.tokenize())
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_integers(self):
        """Test integer literals."""
        lexer = Lexer("42 123 0")
        tokens = list(lexer.tokenize())
        
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == 42
        
        assert tokens[1].type == TokenType.INTEGER
        assert tokens[1].value == 123
        
        assert tokens[2].type == TokenType.INTEGER
        assert tokens[2].value == 0
    
    def test_floats(self):
        """Test float literals."""
        lexer = Lexer("3.14 0.5 1e10 2.5e-3")
        tokens = list(lexer.tokenize())
        
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == 3.14
        
        assert tokens[1].type == TokenType.FLOAT
        assert tokens[1].value == 0.5
    
    def test_strings(self):
        """Test string literals."""
        lexer = Lexer('"hello" \'world\'')
        tokens = list(lexer.tokenize())
        
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"
        
        assert tokens[1].type == TokenType.STRING
        assert tokens[1].value == "world"
    
    def test_triple_strings(self):
        """Test triple-quoted strings."""
        source = '''"""multi
line
string"""'''
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        assert tokens[0].type == TokenType.TRIPLE_STRING
        assert "multi" in tokens[0].value
        assert "line" in tokens[0].value
    
    def test_keywords(self):
        """Test keyword recognition."""
        lexer = Lexer("fn let if else for while return pipeline")
        tokens = list(lexer.tokenize())
        
        assert tokens[0].type == TokenType.FN
        assert tokens[1].type == TokenType.LET
        assert tokens[2].type == TokenType.IF
        assert tokens[3].type == TokenType.ELSE
        assert tokens[4].type == TokenType.FOR
        assert tokens[5].type == TokenType.WHILE
        assert tokens[6].type == TokenType.RETURN
        assert tokens[7].type == TokenType.PIPELINE
    
    def test_operators(self):
        """Test operator tokenization."""
        lexer = Lexer("+ - * / == != < > <= >=")
        tokens = list(lexer.tokenize())
        
        expected = [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT,
            TokenType.LE, TokenType.GE,
        ]
        
        for token, expected_type in zip(tokens[:-1], expected):
            assert token.type == expected_type
    
    def test_delimiters(self):
        """Test delimiter tokenization."""
        lexer = Lexer("( ) [ ] { } , : ; . ->")
        tokens = list(lexer.tokenize())
        
        expected = [
            TokenType.LPAREN, TokenType.RPAREN,
            TokenType.LBRACKET, TokenType.RBRACKET,
            TokenType.LBRACE, TokenType.RBRACE,
            TokenType.COMMA, TokenType.COLON, TokenType.SEMICOLON,
            TokenType.DOT, TokenType.ARROW,
        ]
        
        for token, expected_type in zip(tokens[:-1], expected):
            assert token.type == expected_type
    
    def test_annotations(self):
        """Test annotation tokenization."""
        lexer = Lexer("@intent @model @hardware @natural_language")
        tokens = list(lexer.tokenize())
        
        assert tokens[0].type == TokenType.INTENT
        assert tokens[1].type == TokenType.MODEL
        assert tokens[2].type == TokenType.HARDWARE
        assert tokens[3].type == TokenType.NATURAL_LANGUAGE
    
    def test_comments(self):
        """Test comment handling."""
        lexer = Lexer("42 # this is a comment\n43")
        tokens = list(lexer.tokenize())
        
        # Should have: 42, NEWLINE, 43, EOF
        int_tokens = [t for t in tokens if t.type == TokenType.INTEGER]
        assert len(int_tokens) == 2
        assert int_tokens[0].value == 42
        assert int_tokens[1].value == 43
    
    def test_indentation(self):
        """Test indentation handling."""
        source = """fn test():
    pass"""
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        token_types = [t.type for t in tokens]
        assert TokenType.INDENT in token_types
        assert TokenType.DEDENT in token_types
    
    def test_function_definition(self):
        """Test tokenizing a function definition."""
        source = """fn add(a: Int, b: Int) -> Int:
    return a + b"""
        
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        
        # Check key tokens
        assert any(t.type == TokenType.FN for t in tokens)
        assert any(t.type == TokenType.IDENTIFIER and t.value == "add" for t in tokens)
        assert any(t.type == TokenType.RETURN for t in tokens)


class TestTokenMethods:
    """Test Token class methods."""
    
    def test_is_literal(self):
        """Test is_literal method."""
        from trident.lexer.tokens import SourceLocation
        
        loc = SourceLocation(1, 1)
        
        int_token = Token(TokenType.INTEGER, 42, loc, "42")
        assert int_token.is_literal()
        
        id_token = Token(TokenType.IDENTIFIER, "foo", loc, "foo")
        assert not id_token.is_literal()
    
    def test_is_operator(self):
        """Test is_operator method."""
        from trident.lexer.tokens import SourceLocation
        
        loc = SourceLocation(1, 1)
        
        plus_token = Token(TokenType.PLUS, "+", loc, "+")
        assert plus_token.is_operator()
        
        id_token = Token(TokenType.IDENTIFIER, "foo", loc, "foo")
        assert not id_token.is_operator()
