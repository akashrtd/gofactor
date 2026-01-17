"""
Abstract Syntax Tree (AST) node definitions for Trident.

This module defines all AST nodes used to represent Trident programs.
The design follows these principles:

1. IMMUTABLE: All nodes are frozen dataclasses for safety
2. TYPED: Full type hints for static analysis
3. VISITABLE: Support visitor pattern for tree traversal
4. LOCATED: All nodes carry source location for error reporting

AST Structure:
    Program
    ├── Annotations (decorators)
    ├── Imports
    ├── PipelineDef
    │   ├── Annotations
    │   └── FunctionDef[]
    └── FunctionDef
        ├── Parameters
        └── Statement[]
            ├── LetStatement
            ├── ReturnStatement
            ├── IfStatement
            ├── ForStatement
            └── ExpressionStmt
                └── Expression
                    ├── Literal
                    ├── Identifier
                    ├── BinaryOp
                    ├── UnaryOp
                    ├── CallExpr
                    ├── IndexExpr
                    ├── AttributeExpr
                    ├── TensorExpr
                    └── NaturalLanguageExpr
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from enum import Enum, auto

from trident.lexer.tokens import SourceLocation, Token


# =============================================================================
# Base Classes
# =============================================================================

@dataclass(frozen=True)
class ASTNode(ABC):
    """Base class for all AST nodes."""
    location: SourceLocation
    
    @abstractmethod
    def accept(self, visitor: "ASTVisitor") -> Any:
        """Accept a visitor for tree traversal."""
        pass
    
    def __str__(self) -> str:
        return self.__class__.__name__


class ASTVisitor(ABC):
    """Base visitor for AST traversal."""
    
    def visit(self, node: ASTNode) -> Any:
        """Visit a node by dispatching to the appropriate method."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor_method = getattr(self, method_name, self.generic_visit)
        return visitor_method(node)
    
    def generic_visit(self, node: ASTNode) -> Any:
        """Default visitor for unhandled nodes."""
        raise NotImplementedError(f"No visitor for {node.__class__.__name__}")


# =============================================================================
# Type Annotations
# =============================================================================

@dataclass(frozen=True)
class TypeAnnotation(ASTNode):
    """Base class for type annotations."""
    pass


@dataclass(frozen=True)
class SimpleType(TypeAnnotation):
    """Simple type like Int, String, Float, Bool."""
    name: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class TensorType(TypeAnnotation):
    """Tensor type with optional shape: Tensor[Float, 32, 64]."""
    dtype: str
    shape: tuple[int | str, ...] = ()
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class GenericType(TypeAnnotation):
    """Generic type like List[Int] or Dict[String, Float]."""
    name: str
    params: tuple[TypeAnnotation, ...]
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class FunctionType(TypeAnnotation):
    """Function type: (Int, Int) -> Int."""
    params: tuple[TypeAnnotation, ...]
    return_type: TypeAnnotation
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


# =============================================================================
# Expressions
# =============================================================================

@dataclass(frozen=True)
class Expression(ASTNode):
    """Base class for all expressions."""
    pass


@dataclass(frozen=True)
class Literal(Expression):
    """Literal values: integers, floats, strings, booleans."""
    value: int | float | str | bool | None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class Identifier(Expression):
    """Variable or function name reference."""
    name: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


class BinaryOperator(Enum):
    """Binary operators."""
    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FLOOR_DIV = auto()
    MOD = auto()
    POW = auto()
    MATMUL = auto()
    
    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    
    # Logical
    AND = auto()
    OR = auto()


class UnaryOperator(Enum):
    """Unary operators."""
    NEG = auto()
    NOT = auto()
    POS = auto()


@dataclass(frozen=True)
class BinaryOp(Expression):
    """Binary operation: left op right."""
    left: Expression
    operator: BinaryOperator
    right: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class UnaryOp(Expression):
    """Unary operation: op operand."""
    operator: UnaryOperator
    operand: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class CallExpr(Expression):
    """Function or method call: func(args)."""
    callee: Expression
    arguments: tuple[Expression, ...]
    keyword_args: tuple[tuple[str, Expression], ...] = ()
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class IndexExpr(Expression):
    """Index/subscript access: obj[index]."""
    object: Expression
    index: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class AttributeExpr(Expression):
    """Attribute access: obj.attr."""
    object: Expression
    attribute: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class TensorExpr(Expression):
    """Tensor literal: [1, 2, 3] or [[1, 2], [3, 4]]."""
    elements: tuple[Expression, ...]
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class DictExpr(Expression):
    """Dictionary literal: {key: value}."""
    items: tuple[tuple[Expression, Expression], ...]
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class NaturalLanguageExpr(Expression):
    """
    Natural language block for LLM queries.
    
    @natural_language
    result = llm.query(context, \"\"\"
        Extract the vendor name and total amount.
    \"\"\")
    """
    content: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class LambdaExpr(Expression):
    """Lambda expression: |x, y| x + y."""
    params: tuple[str, ...]
    body: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class IfExpr(Expression):
    """Conditional expression: if cond then a else b."""
    condition: Expression
    then_expr: Expression
    else_expr: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


# =============================================================================
# Statements
# =============================================================================

@dataclass(frozen=True)
class Statement(ASTNode):
    """Base class for all statements."""
    pass


@dataclass(frozen=True)
class ExpressionStmt(Statement):
    """Expression used as a statement."""
    expression: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class LetStatement(Statement):
    """Variable declaration: let x = expr or let x: Type = expr."""
    name: str
    value: Expression
    type_annotation: Optional[TypeAnnotation] = None
    is_const: bool = False
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class AssignmentStmt(Statement):
    """Assignment statement: x = expr."""
    target: Expression  # Can be Identifier, IndexExpr, or AttributeExpr
    value: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class ReturnStatement(Statement):
    """Return statement: return expr."""
    value: Optional[Expression] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class IfStatement(Statement):
    """If statement with optional elif and else branches."""
    condition: Expression
    then_body: tuple[Statement, ...]
    elif_branches: tuple[tuple[Expression, tuple[Statement, ...]], ...] = ()
    else_body: tuple[Statement, ...] = ()
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class ForStatement(Statement):
    """For loop: for x in iterable: body."""
    variable: str
    iterable: Expression
    body: tuple[Statement, ...]
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class WhileStatement(Statement):
    """While loop: while condition: body."""
    condition: Expression
    body: tuple[Statement, ...]
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class PassStatement(Statement):
    """Pass statement (no-op)."""
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class BreakStatement(Statement):
    """Break statement."""
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class ContinueStatement(Statement):
    """Continue statement."""
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class ImportStatement(Statement):
    """Import statement: import module or from module import name."""
    module: str
    names: tuple[tuple[str, Optional[str]], ...] = ()  # (name, alias) pairs
    is_from_import: bool = False
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


# =============================================================================
# Annotations (Decorators)
# =============================================================================

@dataclass(frozen=True)
class Annotation(ASTNode):
    """
    Annotation/decorator: @intent("..."), @model(...), @hardware(...).
    """
    name: str
    arguments: tuple[Expression, ...] = ()
    keyword_args: tuple[tuple[str, Expression], ...] = ()
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


# =============================================================================
# Definitions
# =============================================================================

@dataclass(frozen=True)
class Parameter(ASTNode):
    """Function parameter with optional type and default value."""
    name: str
    type_annotation: Optional[TypeAnnotation] = None
    default_value: Optional[Expression] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class FunctionDef(Statement):
    """Function definition: fn name(params) -> ReturnType: body."""
    name: str
    params: tuple[Parameter, ...]
    body: tuple[Statement, ...]
    return_type: Optional[TypeAnnotation] = None
    annotations: tuple[Annotation, ...] = ()
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class PipelineDef(Statement):
    """
    Pipeline definition with AI annotations.
    
    @intent("...")
    @model(ocr: "...", nlp: "...")
    pipeline Name:
        fn process(...): ...
    """
    name: str
    body: tuple[Statement, ...]  # Usually FunctionDefs
    annotations: tuple[Annotation, ...] = ()
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class Field(ASTNode):
    """Struct field: name: Type = default."""
    name: str
    type_annotation: TypeAnnotation
    default_value: Optional[Expression] = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


@dataclass(frozen=True)
class StructDef(Statement):
    """
    Struct definition for data shapes.
    
    struct Name:
        field: Type
    """
    name: str
    fields: tuple[Field, ...]

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)


# =============================================================================
# Program (Root Node)
# =============================================================================

@dataclass(frozen=True)
class Program(ASTNode):
    """Root node of the AST - represents an entire Trident file."""
    statements: tuple[Statement, ...]
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit(self)
