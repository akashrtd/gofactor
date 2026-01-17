"""
Semantic Analyzer for the Trident programming language.

Performs:
1. Scope resolution - Binding names to declarations
2. Type checking - Verifying type consistency
3. Type inference - Deducing types where not specified
4. Validation - Checking semantic constraints
"""

from dataclasses import dataclass, field
from typing import Optional

from trident.parser.ast import (
    ASTNode,
    ASTVisitor,
    Program,
    Statement,
    Expression,
    # Statements
    ExpressionStmt,
    LetStatement,
    AssignmentStmt,
    ReturnStatement,
    IfStatement,
    ForStatement,
    WhileStatement,
    PassStatement,
    BreakStatement,
    ContinueStatement,
    ImportStatement,
    FunctionDef,
    PipelineDef,
    StructDef,
    Parameter,
    Field,
    # Expressions
    Literal,
    Identifier,
    BinaryOp,
    BinaryOperator,
    UnaryOp,
    UnaryOperator,
    CallExpr,
    IndexExpr,
    AttributeExpr,
    TensorExpr,
    DictExpr,
    NaturalLanguageExpr,
    LambdaExpr,
    IfExpr,
    # Annotations
    Annotation,
    # Types
    TypeAnnotation,
    SimpleType,
    TensorType as ASTTensorType,
    GenericType,
)
from trident.semantic.types import (
    Type,
    PrimitiveType,
    TensorType,
    FunctionType,
    PipelineType,
    ModelType,
    ListType,
    DictType,
    AnyType,
    INT, FLOAT, STRING, BOOL, IMAGE, DOCUMENT, ANY, NONE, UNKNOWN,
    type_from_string,
    common_type,
)
from trident.semantic.symbols import Symbol, SymbolTable, SymbolKind, Scope
from trident.lexer.tokens import SourceLocation


class SemanticError(Exception):
    """Exception raised for semantic errors."""
    
    def __init__(self, message: str, location: SourceLocation) -> None:
        self.message = message
        self.location = location
        super().__init__(f"{location}: {message}")


@dataclass
class SemanticAnalyzer(ASTVisitor):
    """
    Semantic analyzer that traverses the AST and performs:
    - Name resolution
    - Type checking
    - Type inference
    
    Usage:
        analyzer = SemanticAnalyzer()
        analyzer.analyze(ast)
        # Access analyzer.errors for any errors found
    """
    
    symbols: SymbolTable = field(default_factory=SymbolTable)
    errors: list[SemanticError] = field(default_factory=list)
    
    # Context tracking
    _current_function: Optional[FunctionDef] = field(default=None, init=False)
    _current_pipeline: Optional[PipelineDef] = field(default=None, init=False)
    _in_loop: bool = field(default=False, init=False)
    _expected_return_type: Optional[Type] = field(default=None, init=False)
    
    # Type annotations for nodes (node id -> type)
    _node_types: dict[int, Type] = field(default_factory=dict, init=False)
    
    def analyze(self, program: Program) -> bool:
        """
        Analyze the program AST.
        
        Returns:
            True if no errors, False otherwise
        """
        self.visit(program)
        return len(self.errors) == 0
    
    def get_type(self, node: ASTNode) -> Type:
        """Get the inferred type for a node."""
        return self._node_types.get(id(node), UNKNOWN)
    
    def _set_type(self, node: ASTNode, typ: Type) -> None:
        """Set the type for a node."""
        self._node_types[id(node)] = typ
    
    def _error(self, message: str, location: SourceLocation) -> None:
        """Record a semantic error."""
        self.errors.append(SemanticError(message, location))
    
    def _resolve_type_annotation(self, ann: TypeAnnotation) -> Type:
        """Convert an AST type annotation to a Type."""
    def _resolve_type_annotation(self, ann: TypeAnnotation) -> Type:
        """Convert an AST type annotation to a Type."""
        if isinstance(ann, SimpleType):
            # Try primitive/builtin types first
            typ = type_from_string(ann.name)
            if typ != UNKNOWN:
                return typ
            
            # Lookup user-defined types (Structs)
            symbol = self.symbols.lookup(ann.name)
            if symbol and symbol.kind == SymbolKind.STRUCT:
                return symbol.type
                
            return UNKNOWN
        elif isinstance(ann, ASTTensorType):
            dtype = type_from_string(ann.dtype) if ann.dtype else FLOAT
            return TensorType(dtype=dtype, shape=ann.shape)
        elif isinstance(ann, GenericType):
            if ann.name == "List":
                elem_type = self._resolve_type_annotation(ann.params[0]) if ann.params else ANY
                return ListType(element_type=elem_type)
            elif ann.name == "Tensor":
                dtype = self._resolve_type_annotation(ann.params[0]) if ann.params else FLOAT
                return TensorType(dtype=dtype)
            return ANY
        return UNKNOWN
    
    # =========================================================================
    # Visitor Methods
    # =========================================================================
    
    def visit_Program(self, node: Program) -> None:
        """Visit program root."""
        for stmt in node.statements:
            self.visit(stmt)
    
    def visit_PipelineDef(self, node: PipelineDef) -> None:
        """Visit pipeline definition."""
        location = node.location
        
        # Check for redefinition
        if self.symbols.lookup_local(node.name):
            self._error(f"Redefinition of '{node.name}'", location)
            return
        
        # Create pipeline type
        pipeline_type = PipelineType(name=node.name)
        
        # Define pipeline symbol
        symbol = Symbol(
            name=node.name,
            kind=SymbolKind.PIPELINE,
            type=pipeline_type,
            location=location,
        )
        self.symbols.define(symbol)
        
        # Enter pipeline scope
        old_pipeline = self._current_pipeline
        self._current_pipeline = node
        self.symbols.enter_scope(f"pipeline:{node.name}")
        
        # Process annotations
        for ann in node.annotations:
            self._process_annotation(ann)
        
        # Analyze body
        for stmt in node.body:
            self.visit(stmt)
        
        # Exit scope
        self.symbols.exit_scope()
        self._current_pipeline = old_pipeline
    
    def visit_StructDef(self, node: StructDef) -> None:
        """Visit struct definition."""
        location = node.location
        
        # Check for redefinition
        if self.symbols.lookup_local(node.name):
            self._error(f"Redefinition of '{node.name}'", location)
            return
            
        # Define struct symbol (using AnyType for now as placeholder for StructType)
        symbol = Symbol(
            name=node.name,
            kind=SymbolKind.STRUCT,
            type=ANY, 
            location=location,
        )
        self.symbols.define(symbol)
        
        # Analyze fields
        for field in node.fields:
            self.visit(field)
            
    def visit_Field(self, node: Field) -> None:
        """Visit struct field."""
        if node.default_value:
            self.visit(node.default_value)
            value_type = self.get_type(node.default_value)
            declared_type = self._resolve_type_annotation(node.type_annotation)
            
            if not declared_type.is_assignable_from(value_type):
                self._error(
                    f"Cannot assign {value_type} to field {node.name}:{declared_type}",
                    node.location
                )
    
    def visit_FunctionDef(self, node: FunctionDef) -> None:
        """Visit function definition."""
        location = node.location
        
        # Check for redefinition in current scope
        if self.symbols.lookup_local(node.name):
            self._error(f"Redefinition of '{node.name}'", location)
            return
        
        # Resolve parameter types
        param_types: list[Type] = []
        param_names: list[str] = []
        
        for param in node.params:
            if param.type_annotation:
                param_type = self._resolve_type_annotation(param.type_annotation)
            else:
                param_type = ANY  # Gradual typing
            param_types.append(param_type)
            param_names.append(param.name)
        
        # Resolve return type
        return_type = ANY
        if node.return_type:
            return_type = self._resolve_type_annotation(node.return_type)
        
        # Create function type
        func_type = FunctionType(
            param_types=tuple(param_types),
            return_type=return_type,
            param_names=tuple(param_names),
        )
        
        # Define function symbol
        symbol = Symbol(
            name=node.name,
            kind=SymbolKind.FUNCTION,
            type=func_type,
            location=location,
            is_mutable=False,
        )
        self.symbols.define(symbol)
        
        # Enter function scope
        old_function = self._current_function
        old_return_type = self._expected_return_type
        self._current_function = node
        self._expected_return_type = return_type
        self.symbols.enter_scope(f"fn:{node.name}")
        
        # Define parameters in function scope
        for param, param_type in zip(node.params, param_types):
            param_symbol = Symbol(
                name=param.name,
                kind=SymbolKind.PARAMETER,
                type=param_type,
                location=param.location,
                is_mutable=True,
            )
            self.symbols.define(param_symbol)
        
        # Process annotations
        for ann in node.annotations:
            self._process_annotation(ann)
        
        # Analyze body
        for stmt in node.body:
            self.visit(stmt)
        
        # Exit scope
        self.symbols.exit_scope()
        self._current_function = old_function
        self._expected_return_type = old_return_type
        
        self._set_type(node, func_type)
    
    def visit_LetStatement(self, node: LetStatement) -> None:
        """Visit let/const statement."""
        location = node.location
        
        # Check for redefinition
        if self.symbols.lookup_local(node.name):
            self._error(f"Redefinition of '{node.name}'", location)
            return
        
        # Analyze value expression
        self.visit(node.value)
        value_type = self.get_type(node.value)
        
        # Determine declared type
        declared_type: Type
        if node.type_annotation:
            declared_type = self._resolve_type_annotation(node.type_annotation)
            # Check type compatibility
            if not declared_type.is_assignable_from(value_type):
                self._error(
                    f"Cannot assign {value_type} to {declared_type}",
                    location
                )
        else:
            declared_type = value_type if value_type != UNKNOWN else ANY
        
        # Define symbol
        symbol = Symbol(
            name=node.name,
            kind=SymbolKind.CONSTANT if node.is_const else SymbolKind.VARIABLE,
            type=declared_type,
            location=location,
            is_mutable=not node.is_const,
        )
        self.symbols.define(symbol)
    
    def visit_AssignmentStmt(self, node: AssignmentStmt) -> None:
        """Visit assignment statement."""
        # Analyze target and value
        self.visit(node.target)
        self.visit(node.value)
        
        target_type = self.get_type(node.target)
        value_type = self.get_type(node.value)
        
        # Check if target is assignable
        if isinstance(node.target, Identifier):
            symbol = self.symbols.lookup(node.target.name)
            if symbol:
                if not symbol.is_mutable:
                    self._error(
                        f"Cannot assign to constant '{node.target.name}'",
                        node.location
                    )
                if not symbol.type.is_assignable_from(value_type):
                    self._error(
                        f"Cannot assign {value_type} to {symbol.type}",
                        node.location
                    )
        
        # Type check
        if not target_type.is_assignable_from(value_type):
            self._error(
                f"Cannot assign {value_type} to {target_type}",
                node.location
            )
    
    def visit_ReturnStatement(self, node: ReturnStatement) -> None:
        """Visit return statement."""
        if self._current_function is None:
            self._error("Return statement outside function", node.location)
            return
        
        if node.value:
            self.visit(node.value)
            return_type = self.get_type(node.value)
        else:
            return_type = NONE
        
        # Check return type
        if self._expected_return_type and self._expected_return_type != ANY:
            if not self._expected_return_type.is_assignable_from(return_type):
                self._error(
                    f"Expected return type {self._expected_return_type}, got {return_type}",
                    node.location
                )
    
    def visit_IfStatement(self, node: IfStatement) -> None:
        """Visit if statement."""
        # Check condition
        self.visit(node.condition)
        cond_type = self.get_type(node.condition)
        
        if cond_type != BOOL and cond_type != ANY and cond_type != UNKNOWN:
            self._error(f"Condition must be Bool, got {cond_type}", node.location)
        
        # Analyze then branch
        self.symbols.enter_scope("if:then")
        for stmt in node.then_body:
            self.visit(stmt)
        self.symbols.exit_scope()
        
        # Analyze elif branches
        for elif_cond, elif_body in node.elif_branches:
            self.visit(elif_cond)
            self.symbols.enter_scope("if:elif")
            for stmt in elif_body:
                self.visit(stmt)
            self.symbols.exit_scope()
        
        # Analyze else branch
        if node.else_body:
            self.symbols.enter_scope("if:else")
            for stmt in node.else_body:
                self.visit(stmt)
            self.symbols.exit_scope()
    
    def visit_ForStatement(self, node: ForStatement) -> None:
        """Visit for loop."""
        # Analyze iterable
        self.visit(node.iterable)
        
        # Enter loop scope
        old_in_loop = self._in_loop
        self._in_loop = True
        self.symbols.enter_scope("for")
        
        # Define loop variable
        # TODO: Infer element type from iterable
        loop_var = Symbol(
            name=node.variable,
            kind=SymbolKind.VARIABLE,
            type=ANY,
            location=node.location,
        )
        self.symbols.define(loop_var)
        
        # Analyze body
        for stmt in node.body:
            self.visit(stmt)
        
        self.symbols.exit_scope()
        self._in_loop = old_in_loop
    
    def visit_WhileStatement(self, node: WhileStatement) -> None:
        """Visit while loop."""
        # Check condition
        self.visit(node.condition)
        cond_type = self.get_type(node.condition)
        
        if cond_type != BOOL and cond_type != ANY and cond_type != UNKNOWN:
            self._error(f"Condition must be Bool, got {cond_type}", node.location)
        
        # Enter loop scope
        old_in_loop = self._in_loop
        self._in_loop = True
        self.symbols.enter_scope("while")
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.symbols.exit_scope()
        self._in_loop = old_in_loop
    
    def visit_BreakStatement(self, node: BreakStatement) -> None:
        """Visit break statement."""
        if not self._in_loop:
            self._error("Break statement outside loop", node.location)
    
    def visit_ContinueStatement(self, node: ContinueStatement) -> None:
        """Visit continue statement."""
        if not self._in_loop:
            self._error("Continue statement outside loop", node.location)
    
    def visit_PassStatement(self, node: PassStatement) -> None:
        """Visit pass statement (no-op)."""
        pass
    
    def visit_ImportStatement(self, node: ImportStatement) -> None:
        """Visit import statement."""
        # TODO: Module resolution
        if node.is_from_import:
            for name, alias in node.names:
                symbol_name = alias or name
                self.symbols.define(Symbol(
                    name=symbol_name,
                    kind=SymbolKind.MODULE,
                    type=ANY,
                    location=node.location,
                ))
        else:
            self.symbols.define(Symbol(
                name=node.module,
                kind=SymbolKind.MODULE,
                type=ANY,
                location=node.location,
            ))
    
    def visit_ExpressionStmt(self, node: ExpressionStmt) -> None:
        """Visit expression statement."""
        self.visit(node.expression)
    
    # =========================================================================
    # Expression Visitors
    # =========================================================================
    
    def visit_Literal(self, node: Literal) -> None:
        """Visit literal expression."""
        if isinstance(node.value, bool):
            self._set_type(node, BOOL)
        elif isinstance(node.value, int):
            self._set_type(node, INT)
        elif isinstance(node.value, float):
            self._set_type(node, FLOAT)
        elif isinstance(node.value, str):
            self._set_type(node, STRING)
        elif node.value is None:
            self._set_type(node, NONE)
        else:
            self._set_type(node, ANY)
    
    def visit_Identifier(self, node: Identifier) -> None:
        """Visit identifier expression."""
        symbol = self.symbols.lookup(node.name)
        if symbol is None:
            self._error(f"Undefined name '{node.name}'", node.location)
            self._set_type(node, UNKNOWN)
        else:
            self._set_type(node, symbol.type)
    
    def visit_BinaryOp(self, node: BinaryOp) -> None:
        """Visit binary operation."""
        self.visit(node.left)
        self.visit(node.right)
        
        left_type = self.get_type(node.left)
        right_type = self.get_type(node.right)
        
        op = node.operator
        
        # Comparison operators always return Bool
        if op in (BinaryOperator.EQ, BinaryOperator.NE, BinaryOperator.LT,
                  BinaryOperator.GT, BinaryOperator.LE, BinaryOperator.GE):
            self._set_type(node, BOOL)
            return
        
        # Logical operators
        if op in (BinaryOperator.AND, BinaryOperator.OR):
            self._set_type(node, BOOL)
            return
        
        # Arithmetic operators
        if op in (BinaryOperator.ADD, BinaryOperator.SUB, BinaryOperator.MUL,
                  BinaryOperator.DIV, BinaryOperator.MOD, BinaryOperator.POW,
                  BinaryOperator.FLOOR_DIV):
            # Numeric promotion
            if left_type.is_numeric() and right_type.is_numeric():
                result_type = common_type(left_type, right_type)
                self._set_type(node, result_type)
                return
            
            # Tensor operations
            if left_type.is_tensor_like() or right_type.is_tensor_like():
                self._set_type(node, TensorType())
                return
            
            # String concatenation
            if op == BinaryOperator.ADD and left_type == STRING:
                self._set_type(node, STRING)
                return
        
        # Matrix multiplication
        if op == BinaryOperator.MATMUL:
            if left_type.is_tensor_like() and right_type.is_tensor_like():
                self._set_type(node, TensorType())
                return
        
        self._set_type(node, ANY)
    
    def visit_UnaryOp(self, node: UnaryOp) -> None:
        """Visit unary operation."""
        self.visit(node.operand)
        operand_type = self.get_type(node.operand)
        
        if node.operator == UnaryOperator.NOT:
            self._set_type(node, BOOL)
        elif node.operator in (UnaryOperator.NEG, UnaryOperator.POS):
            self._set_type(node, operand_type)
        else:
            self._set_type(node, operand_type)
    
    def visit_CallExpr(self, node: CallExpr) -> None:
        """Visit function call."""
        self.visit(node.callee)
        callee_type = self.get_type(node.callee)
        
        # Analyze arguments
        for arg in node.arguments:
            self.visit(arg)
        for _, value in node.keyword_args:
            self.visit(value)
        
        if isinstance(callee_type, FunctionType):
            # Check argument count
            if len(node.arguments) != len(callee_type.param_types):
                self._error(
                    f"Expected {len(callee_type.param_types)} arguments, "
                    f"got {len(node.arguments)}",
                    node.location
                )
            
            # Type check arguments
            for arg, param_type in zip(node.arguments, callee_type.param_types):
                arg_type = self.get_type(arg)
                if not param_type.is_assignable_from(arg_type):
                    self._error(
                        f"Argument type mismatch: expected {param_type}, got {arg_type}",
                        arg.location
                    )
            
            self._set_type(node, callee_type.return_type)
        else:
            # Unknown callable
            self._set_type(node, ANY)
    
    def visit_IndexExpr(self, node: IndexExpr) -> None:
        """Visit index expression."""
        self.visit(node.object)
        self.visit(node.index)
        
        obj_type = self.get_type(node.object)
        
        if isinstance(obj_type, TensorType):
            # Indexing a tensor
            self._set_type(node, obj_type.dtype)
        elif isinstance(obj_type, ListType):
            self._set_type(node, obj_type.element_type)
        else:
            self._set_type(node, ANY)
    
    def visit_AttributeExpr(self, node: AttributeExpr) -> None:
        """Visit attribute access."""
        self.visit(node.object)
        obj_type = self.get_type(node.object)
        
        # TODO: Implement proper attribute lookup
        # For now, return Any
        self._set_type(node, ANY)
    
    def visit_TensorExpr(self, node: TensorExpr) -> None:
        """Visit tensor literal."""
        element_types: list[Type] = []
        
        for elem in node.elements:
            self.visit(elem)
            element_types.append(self.get_type(elem))
        
        # Determine common element type
        if element_types:
            dtype = element_types[0]
            for t in element_types[1:]:
                dtype = common_type(dtype, t)
            self._set_type(node, TensorType(dtype=dtype, shape=(len(element_types),)))
        else:
            self._set_type(node, TensorType())
            
    def visit_DictExpr(self, node: DictExpr) -> None:
        """Visit dictionary literal."""
        key_types: list[Type] = []
        value_types: list[Type] = []
        
        for key, value in node.items:
            self.visit(key)
            self.visit(value)
            key_types.append(self.get_type(key))
            value_types.append(self.get_type(value))
            
        # Determine common key and value types
        if key_types:
            key_dtype = key_types[0]
            for t in key_types[1:]:
                key_dtype = common_type(key_dtype, t)
        else:
            key_dtype = STRING # Default to String keys
            
        if value_types:
            val_dtype = value_types[0]
            for t in value_types[1:]:
                val_dtype = common_type(val_dtype, t)
        else:
            val_dtype = ANY
            
        self._set_type(node, DictType(key_type=key_dtype, value_type=val_dtype))
    
    def visit_NaturalLanguageExpr(self, node: NaturalLanguageExpr) -> None:
        """Visit natural language expression."""
        # Natural language blocks return dynamic/any type
        self._set_type(node, ANY)
    
    def visit_LambdaExpr(self, node: LambdaExpr) -> None:
        """Visit lambda expression."""
        # Enter lambda scope
        self.symbols.enter_scope("lambda")
        
        # Define parameters
        param_types: list[Type] = []
        for param in node.params:
            symbol = Symbol(
                name=param,
                kind=SymbolKind.PARAMETER,
                type=ANY,
                location=node.location,
            )
            self.symbols.define(symbol)
            param_types.append(ANY)
        
        # Analyze body
        self.visit(node.body)
        return_type = self.get_type(node.body)
        
        self.symbols.exit_scope()
        
        self._set_type(node, FunctionType(
            param_types=tuple(param_types),
            return_type=return_type,
        ))
    
    def visit_IfExpr(self, node: IfExpr) -> None:
        """Visit conditional expression."""
        self.visit(node.condition)
        self.visit(node.then_expr)
        self.visit(node.else_expr)
        
        then_type = self.get_type(node.then_expr)
        else_type = self.get_type(node.else_expr)
        
        result_type = common_type(then_type, else_type)
        self._set_type(node, result_type)
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _process_annotation(self, ann: Annotation) -> None:
        """Process an annotation and record model/hardware info."""
        # @model annotations define model bindings
        if ann.name == "model":
            for name, value in ann.keyword_args:
                if isinstance(value, Literal) and isinstance(value.value, str):
                    model_type = ModelType(
                        name=value.value,
                        category=name,
                    )
                    self.symbols.define(Symbol(
                        name=f"__{name}_model__",
                        kind=SymbolKind.MODEL,
                        type=model_type,
                        location=ann.location,
                    ))


def analyze(program: Program) -> tuple[bool, list[SemanticError], SemanticAnalyzer]:
    """
    Convenience function to analyze a program.
    
    Returns:
        Tuple of (success, errors, analyzer)
    """
    analyzer = SemanticAnalyzer()
    success = analyzer.analyze(program)
    return success, analyzer.errors, analyzer
