"""
Trident Parser - Parsing and AST generation for the Trident programming language.
"""

from trident.parser.ast import (
    # Base
    ASTNode,
    Program,
    # Statements
    Statement,
    ExpressionStmt,
    LetStatement,
    ReturnStatement,
    IfStatement,
    ForStatement,
    WhileStatement,
    PassStatement,
    ImportStatement,
    # Definitions
    FunctionDef,
    PipelineDef,
    Parameter,
    # Expressions
    Expression,
    Literal,
    Identifier,
    BinaryOp,
    UnaryOp,
    CallExpr,
    IndexExpr,
    AttributeExpr,
    TensorExpr,
    NaturalLanguageExpr,
    # Annotations
    Annotation,
)
from trident.parser.parser import Parser, parse, ParseError

__all__ = [
    # Base
    "ASTNode",
    "Program",
    # Statements
    "Statement",
    "ExpressionStmt",
    "LetStatement",
    "ReturnStatement",
    "IfStatement",
    "ForStatement",
    "WhileStatement",
    "PassStatement",
    "ImportStatement",
    # Definitions
    "FunctionDef",
    "PipelineDef",
    "Parameter",
    # Expressions
    "Expression",
    "Literal",
    "Identifier",
    "BinaryOp",
    "UnaryOp",
    "CallExpr",
    "IndexExpr",
    "AttributeExpr",
    "TensorExpr",
    "NaturalLanguageExpr",
    # Annotations
    "Annotation",
    # Parser
    "Parser",
    "parse",
    "ParseError",
]
