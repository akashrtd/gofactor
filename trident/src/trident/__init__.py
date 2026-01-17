"""
Trident - A tri-modal programming language for AI-Human-TPU communication.

This package provides the compiler, runtime, and tools for the Trident language.
"""

__version__ = "0.1.0"
__author__ = "Trident Team"

from trident.lexer import Lexer, Token, TokenType
from trident.parser import Parser, parse
from trident.codegen import compile_to_jax

__all__ = [
    "Lexer",
    "Token", 
    "TokenType",
    "Parser",
    "parse",
    "compile_to_jax",
    "__version__",
]
