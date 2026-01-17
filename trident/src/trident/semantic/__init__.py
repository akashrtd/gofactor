"""
Trident Semantic Analysis - Type checking and scope resolution.
"""

from trident.semantic.types import (
    Type,
    PrimitiveType,
    TensorType,
    FunctionType,
    PipelineType,
    ModelType,
    AnyType,
    NoneType,
    UnknownType,
    INT,
    FLOAT,
    STRING,
    BOOL,
    IMAGE,
    DOCUMENT,
    NONE,
    ANY,
)
from trident.semantic.symbols import Symbol, SymbolTable, Scope
from trident.semantic.analyzer import SemanticAnalyzer, SemanticError, analyze

__all__ = [
    # Types
    "Type",
    "PrimitiveType",
    "TensorType",
    "FunctionType",
    "PipelineType",
    "ModelType",
    "AnyType",
    "NoneType",
    "UnknownType",
    # Type constants
    "INT",
    "FLOAT",
    "STRING",
    "BOOL",
    "IMAGE",
    "DOCUMENT",
    "NONE",
    "ANY",
    # Symbols
    "Symbol",
    "SymbolTable",
    "Scope",
    # Analyzer
    "SemanticAnalyzer",
    "SemanticError",
    "analyze",
]
