"""
Recursive Descent Parser for the Trident programming language.

This parser implements a recursive descent strategy with operator precedence
climbing for expression parsing. It produces an Abstract Syntax Tree (AST)
from a stream of tokens.

Grammar Overview (EBNF):
    program         = statement*
    statement       = pipeline_def | function_def | struct_def | let_stmt | return_stmt
                    | if_stmt | for_stmt | while_stmt | import_stmt
                    | expression_stmt | pass_stmt
    pipeline_def    = annotation* "pipeline" IDENTIFIER ":" NEWLINE INDENT statement+ DEDENT
    struct_def      = "struct" IDENTIFIER ":" NEWLINE INDENT field+ DEDENT
    field           = IDENTIFIER ":" type ("=" expression)? NEWLINE
    pipeline_def    = annotation* "pipeline" IDENTIFIER ":" NEWLINE INDENT statement+ DEDENT
    function_def    = annotation* "fn" IDENTIFIER "(" params? ")" ("->" type)? ":" NEWLINE INDENT statement+ DEDENT
    annotation      = "@" IDENTIFIER ("(" arguments ")")?
    let_stmt        = "let" IDENTIFIER (":" type)? "=" expression NEWLINE
    expression      = or_expr
    or_expr         = and_expr ("or" and_expr)*
    and_expr        = not_expr ("and" not_expr)*
    not_expr        = "not" not_expr | comparison
    comparison      = term (comp_op term)*
    term            = factor (("+"|"-") factor)*
    factor          = unary (("*"|"/"|"//"|"%"|"@") unary)*
    unary           = ("-"|"+"|"not") unary | power
    power           = call ("**" unary)?
    call            = primary ("(" arguments ")" | "[" expression "]" | "." IDENTIFIER)*
    primary         = LITERAL | IDENTIFIER | "(" expression ")" | "[" elements "]"
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import IntEnum

from trident.lexer.tokens import Token, TokenType, SourceLocation
from trident.lexer.lexer import Lexer
from trident.parser.ast import (
    # Base
    ASTNode,
    Program,
    TypeAnnotation,
    SimpleType,
    TensorType,
    GenericType,
    # Statements
    Statement,
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
    # Definitions
    FunctionDef,
    PipelineDef,
    StructDef,
    Parameter,
    Field,
    # Expressions
    Expression,
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
    NaturalLanguageExpr,
    LambdaExpr,
    IfExpr,
    # Annotations
    Annotation,
)


class ParseError(Exception):
    """Exception raised for parsing errors."""
    
    def __init__(self, message: str, location: SourceLocation) -> None:
        self.message = message
        self.location = location
        super().__init__(f"{location}: {message}")


class Precedence(IntEnum):
    """Operator precedence levels (higher = tighter binding)."""
    NONE = 0
    OR = 1
    AND = 2
    NOT = 3
    COMPARISON = 4
    TERM = 5
    FACTOR = 6
    UNARY = 7
    POWER = 8
    CALL = 9
    PRIMARY = 10


# Operator token to binary operator mapping
BINARY_OPS: dict[TokenType, BinaryOperator] = {
    TokenType.PLUS: BinaryOperator.ADD,
    TokenType.MINUS: BinaryOperator.SUB,
    TokenType.STAR: BinaryOperator.MUL,
    TokenType.SLASH: BinaryOperator.DIV,
    TokenType.DOUBLE_SLASH: BinaryOperator.FLOOR_DIV,
    TokenType.PERCENT: BinaryOperator.MOD,
    TokenType.DOUBLE_STAR: BinaryOperator.POW,
    TokenType.MATMUL: BinaryOperator.MATMUL,
    TokenType.AT: BinaryOperator.MATMUL,  # @ can be matmul in expressions
    TokenType.EQ: BinaryOperator.EQ,
    TokenType.NE: BinaryOperator.NE,
    TokenType.LT: BinaryOperator.LT,
    TokenType.GT: BinaryOperator.GT,
    TokenType.LE: BinaryOperator.LE,
    TokenType.GE: BinaryOperator.GE,
    TokenType.AND: BinaryOperator.AND,
    TokenType.OR: BinaryOperator.OR,
}

# Precedence for binary operators
OP_PRECEDENCE: dict[TokenType, Precedence] = {
    TokenType.OR: Precedence.OR,
    TokenType.AND: Precedence.AND,
    TokenType.EQ: Precedence.COMPARISON,
    TokenType.NE: Precedence.COMPARISON,
    TokenType.LT: Precedence.COMPARISON,
    TokenType.GT: Precedence.COMPARISON,
    TokenType.LE: Precedence.COMPARISON,
    TokenType.GE: Precedence.COMPARISON,
    TokenType.PLUS: Precedence.TERM,
    TokenType.MINUS: Precedence.TERM,
    TokenType.STAR: Precedence.FACTOR,
    TokenType.SLASH: Precedence.FACTOR,
    TokenType.DOUBLE_SLASH: Precedence.FACTOR,
    TokenType.PERCENT: Precedence.FACTOR,
    TokenType.AT: Precedence.FACTOR,
    TokenType.DOUBLE_STAR: Precedence.POWER,
}


@dataclass
class Parser:
    """
    Recursive descent parser for Trident.
    
    Usage:
        parser = Parser(tokens)
        ast = parser.parse()
    """
    
    tokens: list[Token]
    filename: str = "<stdin>"
    
    _pos: int = field(default=0, init=False, repr=False)
    _in_natural_language: bool = field(default=False, init=False, repr=False)
    
    @property
    def _current(self) -> Token:
        """Get current token."""
        if self._pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[self._pos]
    
    @property
    def _previous(self) -> Token:
        """Get previous token."""
        return self.tokens[max(0, self._pos - 1)]
    
    def _peek(self, offset: int = 1) -> Token:
        """Peek ahead by offset tokens."""
        pos = self._pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[pos]
    
    def _check(self, *types: TokenType) -> bool:
        """Check if current token is one of the given types."""
        return self._current.type in types
    
    def _match(self, *types: TokenType) -> bool:
        """Check and consume if current token matches."""
        if self._check(*types):
            self._advance()
            return True
        return False
    
    def _advance(self) -> Token:
        """Consume current token and return it."""
        token = self._current
        if token.type != TokenType.EOF:
            self._pos += 1
        return token
    
    def _expect(self, token_type: TokenType, message: str) -> Token:
        """Consume expected token or raise error."""
        if self._check(token_type):
            return self._advance()
        raise ParseError(
            f"{message}, got {self._current.type.name}",
            self._current.location
        )
    
    def _skip_newlines(self) -> None:
        """Skip any newline tokens."""
        while self._check(TokenType.NEWLINE):
            self._advance()
    
    def _error(self, message: str) -> ParseError:
        """Create a parse error at current position."""
        return ParseError(message, self._current.location)
    
    # =========================================================================
    # Top-Level Parsing
    # =========================================================================
    
    def parse(self) -> Program:
        """Parse the entire program and return the AST."""
        statements: list[Statement] = []
        
        self._skip_newlines()
        
        while not self._check(TokenType.EOF):
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
            self._skip_newlines()
        
        location = self.tokens[0].location if self.tokens else SourceLocation(1, 1, self.filename)
        return Program(location=location, statements=tuple(statements))
    
    # =========================================================================
    # Statement Parsing
    # =========================================================================
    
    def _parse_statement(self) -> Optional[Statement]:
        """Parse a single statement."""
        self._skip_newlines()
        
        # Collect annotations
        annotations: list[Annotation] = []
        while self._check(TokenType.INTENT, TokenType.MODEL, TokenType.HARDWARE, 
                          TokenType.NATURAL_LANGUAGE, TokenType.AT):
            annotations.append(self._parse_annotation())
            self._skip_newlines()
        
        # Check for natural language annotation affecting next expression
        if any(a.name == "natural_language" for a in annotations):
            self._in_natural_language = True
        
        # Dispatch based on token type
        if self._check(TokenType.PIPELINE):
            return self._parse_pipeline_def(tuple(annotations))
        elif self._check(TokenType.FN):
            return self._parse_function_def(tuple(annotations))
        elif self._check(TokenType.STRUCT):
            return self._parse_struct_def()
        elif self._check(TokenType.LET, TokenType.CONST):
            return self._parse_let_statement()
        elif self._check(TokenType.RETURN):
            return self._parse_return_statement()
        elif self._check(TokenType.IF):
            return self._parse_if_statement()
        elif self._check(TokenType.FOR):
            return self._parse_for_statement()
        elif self._check(TokenType.WHILE):
            return self._parse_while_statement()
        elif self._check(TokenType.PASS):
            return self._parse_pass_statement()
        elif self._check(TokenType.BREAK):
            return self._parse_break_statement()
        elif self._check(TokenType.CONTINUE):
            return self._parse_continue_statement()
        elif self._check(TokenType.IMPORT, TokenType.FROM):
            return self._parse_import_statement()
        elif self._check(TokenType.NEWLINE, TokenType.EOF, TokenType.DEDENT):
            return None
        else:
            # Expression statement or assignment
            return self._parse_expression_or_assignment()
    
    def _parse_annotation(self) -> Annotation:
        """Parse an annotation: @name or @name(args)."""
        location = self._current.location
        
        # Handle both pre-tokenized annotations and @ followed by identifier
        if self._check(TokenType.AT):
            self._advance()
            name_token = self._expect(TokenType.IDENTIFIER, "Expected annotation name after @")
            name = name_token.value
        else:
            # Pre-tokenized annotation like INTENT, MODEL, etc.
            token = self._advance()
            name = token.value if isinstance(token.value, str) else token.type.name.lower()
        
        # Parse arguments if present
        arguments: list[Expression] = []
        keyword_args: list[tuple[str, Expression]] = []
        
        if self._match(TokenType.LPAREN):
            if not self._check(TokenType.RPAREN):
                self._parse_arguments(arguments, keyword_args)
            self._expect(TokenType.RPAREN, "Expected ')' after annotation arguments")
        
        return Annotation(
            location=location,
            name=name,
            arguments=tuple(arguments),
            keyword_args=tuple(keyword_args),
        )
    
    def _parse_pipeline_def(self, annotations: tuple[Annotation, ...]) -> PipelineDef:
        """Parse a pipeline definition."""
        location = self._current.location
        self._expect(TokenType.PIPELINE, "Expected 'pipeline'")
        
        name_token = self._expect(TokenType.IDENTIFIER, "Expected pipeline name")
        name = name_token.value
        
        self._expect(TokenType.COLON, "Expected ':' after pipeline name")
        self._expect(TokenType.NEWLINE, "Expected newline after ':'")
        self._expect(TokenType.INDENT, "Expected indented block")
        
        body = self._parse_block()
        
        return PipelineDef(
            location=location,
            name=name,
            body=tuple(body),
            annotations=annotations,
        )
    
    def _parse_function_def(self, annotations: tuple[Annotation, ...]) -> FunctionDef:
        """Parse a function definition."""
        location = self._current.location
        self._expect(TokenType.FN, "Expected 'fn'")
        
        name_token = self._expect(TokenType.IDENTIFIER, "Expected function name")
        name = name_token.value
        
        # Parameters
        self._expect(TokenType.LPAREN, "Expected '(' after function name")
        params = self._parse_parameters()
        self._expect(TokenType.RPAREN, "Expected ')' after parameters")
        
        # Return type annotation
        return_type: Optional[TypeAnnotation] = None
        if self._match(TokenType.ARROW):
            return_type = self._parse_type_annotation()
        
        self._expect(TokenType.COLON, "Expected ':' after function signature")
        self._expect(TokenType.NEWLINE, "Expected newline after ':'")
        self._expect(TokenType.INDENT, "Expected indented block")
        
        body = self._parse_block()
        
        return FunctionDef(
            location=location,
            name=name,
            params=tuple(params),
            body=tuple(body),
            return_type=return_type,
            annotations=annotations,
        )
    
    def _parse_struct_def(self) -> StructDef:
        """Parse a struct definition."""
        location = self._current.location
        self._expect(TokenType.STRUCT, "Expected 'struct'")
        
        name_token = self._expect(TokenType.IDENTIFIER, "Expected struct name")
        name = name_token.value
        
        self._expect(TokenType.COLON, "Expected ':' after struct name")
        self._expect(TokenType.NEWLINE, "Expected newline after ':'")
        self._expect(TokenType.INDENT, "Expected indented block")
        
        fields: list[Field] = []
        while not self._check(TokenType.DEDENT, TokenType.EOF):
            field_loc = self._current.location
            field_name_token = self._expect(TokenType.IDENTIFIER, "Expected field name")
            field_name = field_name_token.value
            
            self._expect(TokenType.COLON, "Expected ':' after field name")
            field_type = self._parse_type_annotation()
            
            default_value: Optional[Expression] = None
            if self._match(TokenType.ASSIGN):
                default_value = self._parse_expression()
            
            self._match(TokenType.NEWLINE)
            
            fields.append(Field(
                location=field_loc,
                name=field_name,
                type_annotation=field_type,
                default_value=default_value
            ))
            
            self._skip_newlines()
        
        if self._check(TokenType.DEDENT):
            self._advance()
            
        return StructDef(location=location, name=name, fields=tuple(fields))
    
    def _parse_parameters(self) -> list[Parameter]:
        """Parse function parameters."""
        params: list[Parameter] = []
        
        while not self._check(TokenType.RPAREN):
            location = self._current.location
            name_token = self._expect(TokenType.IDENTIFIER, "Expected parameter name")
            name = name_token.value
            
            # Type annotation
            type_ann: Optional[TypeAnnotation] = None
            if self._match(TokenType.COLON):
                type_ann = self._parse_type_annotation()
            
            # Default value
            default: Optional[Expression] = None
            if self._match(TokenType.ASSIGN):
                default = self._parse_expression()
            
            params.append(Parameter(
                location=location,
                name=name,
                type_annotation=type_ann,
                default_value=default,
            ))
            
            if not self._match(TokenType.COMMA):
                break
        
        return params
    
    def _parse_type_annotation(self) -> TypeAnnotation:
        """Parse a type annotation."""
        location = self._current.location
        
        # Simple type or tensor type
        if self._check(TokenType.IDENTIFIER, TokenType.TENSOR, TokenType.STRING_TYPE,
                       TokenType.INT_TYPE, TokenType.FLOAT_TYPE, TokenType.BOOL_TYPE,
                       TokenType.IMAGE, TokenType.DOCUMENT):
            name_token = self._advance()
            name = name_token.value if isinstance(name_token.value, str) else name_token.type.name
            
            # Check for generic parameters
            if self._match(TokenType.LBRACKET):
                params: list[TypeAnnotation] = []
                while not self._check(TokenType.RBRACKET):
                    params.append(self._parse_type_annotation())
                    if not self._match(TokenType.COMMA):
                        break
                self._expect(TokenType.RBRACKET, "Expected ']' after type parameters")
                return GenericType(location=location, name=name, params=tuple(params))
            
            return SimpleType(location=location, name=name)
        
        raise self._error(f"Expected type annotation, got {self._current.type.name}")
    
    def _parse_block(self) -> list[Statement]:
        """Parse an indented block of statements."""
        statements: list[Statement] = []
        
        while not self._check(TokenType.DEDENT, TokenType.EOF):
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
            self._skip_newlines()
        
        if self._check(TokenType.DEDENT):
            self._advance()
        
        return statements
    
    def _parse_let_statement(self) -> LetStatement:
        """Parse a let/const statement."""
        location = self._current.location
        is_const = self._check(TokenType.CONST)
        self._advance()  # consume let/const
        
        name_token = self._expect(TokenType.IDENTIFIER, "Expected variable name")
        name = name_token.value
        
        # Type annotation
        type_ann: Optional[TypeAnnotation] = None
        if self._match(TokenType.COLON):
            type_ann = self._parse_type_annotation()
        
        self._expect(TokenType.ASSIGN, "Expected '=' after variable name")
        value = self._parse_expression()
        
        # Consume newline if present
        self._match(TokenType.NEWLINE)
        
        return LetStatement(
            location=location,
            name=name,
            value=value,
            type_annotation=type_ann,
            is_const=is_const,
        )
    
    def _parse_return_statement(self) -> ReturnStatement:
        """Parse a return statement."""
        location = self._current.location
        self._expect(TokenType.RETURN, "Expected 'return'")
        
        value: Optional[Expression] = None
        if not self._check(TokenType.NEWLINE, TokenType.DEDENT, TokenType.EOF):
            value = self._parse_expression()
        
        self._match(TokenType.NEWLINE)
        
        return ReturnStatement(location=location, value=value)
    
    def _parse_if_statement(self) -> IfStatement:
        """Parse an if statement with optional elif/else."""
        location = self._current.location
        self._expect(TokenType.IF, "Expected 'if'")
        
        condition = self._parse_expression()
        self._expect(TokenType.COLON, "Expected ':' after if condition")
        self._expect(TokenType.NEWLINE, "Expected newline after ':'")
        self._expect(TokenType.INDENT, "Expected indented block")
        
        then_body = self._parse_block()
        
        # Parse elif branches
        elif_branches: list[tuple[Expression, tuple[Statement, ...]]] = []
        while self._check(TokenType.ELIF):
            self._advance()
            elif_cond = self._parse_expression()
            self._expect(TokenType.COLON, "Expected ':' after elif condition")
            self._expect(TokenType.NEWLINE, "Expected newline after ':'")
            self._expect(TokenType.INDENT, "Expected indented block")
            elif_body = self._parse_block()
            elif_branches.append((elif_cond, tuple(elif_body)))
        
        # Parse else branch
        else_body: list[Statement] = []
        if self._check(TokenType.ELSE):
            self._advance()
            self._expect(TokenType.COLON, "Expected ':' after else")
            self._expect(TokenType.NEWLINE, "Expected newline after ':'")
            self._expect(TokenType.INDENT, "Expected indented block")
            else_body = self._parse_block()
        
        return IfStatement(
            location=location,
            condition=condition,
            then_body=tuple(then_body),
            elif_branches=tuple(elif_branches),
            else_body=tuple(else_body),
        )
    
    def _parse_for_statement(self) -> ForStatement:
        """Parse a for loop."""
        location = self._current.location
        self._expect(TokenType.FOR, "Expected 'for'")
        
        var_token = self._expect(TokenType.IDENTIFIER, "Expected loop variable")
        variable = var_token.value
        
        self._expect(TokenType.IN, "Expected 'in' after loop variable")
        iterable = self._parse_expression()
        
        self._expect(TokenType.COLON, "Expected ':' after iterable")
        self._expect(TokenType.NEWLINE, "Expected newline after ':'")
        self._expect(TokenType.INDENT, "Expected indented block")
        
        body = self._parse_block()
        
        return ForStatement(
            location=location,
            variable=variable,
            iterable=iterable,
            body=tuple(body),
        )
    
    def _parse_while_statement(self) -> WhileStatement:
        """Parse a while loop."""
        location = self._current.location
        self._expect(TokenType.WHILE, "Expected 'while'")
        
        condition = self._parse_expression()
        
        self._expect(TokenType.COLON, "Expected ':' after condition")
        self._expect(TokenType.NEWLINE, "Expected newline after ':'")
        self._expect(TokenType.INDENT, "Expected indented block")
        
        body = self._parse_block()
        
        return WhileStatement(
            location=location,
            condition=condition,
            body=tuple(body),
        )
    
    def _parse_pass_statement(self) -> PassStatement:
        """Parse a pass statement."""
        location = self._current.location
        self._expect(TokenType.PASS, "Expected 'pass'")
        self._match(TokenType.NEWLINE)
        return PassStatement(location=location)
    
    def _parse_break_statement(self) -> BreakStatement:
        """Parse a break statement."""
        location = self._current.location
        self._expect(TokenType.BREAK, "Expected 'break'")
        self._match(TokenType.NEWLINE)
        return BreakStatement(location=location)
    
    def _parse_continue_statement(self) -> ContinueStatement:
        """Parse a continue statement."""
        location = self._current.location
        self._expect(TokenType.CONTINUE, "Expected 'continue'")
        self._match(TokenType.NEWLINE)
        return ContinueStatement(location=location)
    
    def _parse_import_statement(self) -> ImportStatement:
        """Parse an import statement."""
        location = self._current.location
        is_from = self._check(TokenType.FROM)
        
        if is_from:
            self._advance()
            module_token = self._expect(TokenType.IDENTIFIER, "Expected module name")
            module = module_token.value
            
            self._expect(TokenType.IMPORT, "Expected 'import' after module name")
            
            names: list[tuple[str, Optional[str]]] = []
            while True:
                name_token = self._expect(TokenType.IDENTIFIER, "Expected import name")
                name = name_token.value
                alias: Optional[str] = None
                if self._match(TokenType.AS):
                    alias_token = self._expect(TokenType.IDENTIFIER, "Expected alias")
                    alias = alias_token.value
                names.append((name, alias))
                if not self._match(TokenType.COMMA):
                    break
        else:
            self._advance()  # consume 'import'
            module_token = self._expect(TokenType.IDENTIFIER, "Expected module name")
            module = module_token.value
            names = []
            is_from = False
        
        self._match(TokenType.NEWLINE)
        
        return ImportStatement(
            location=location,
            module=module,
            names=tuple(names),
            is_from_import=is_from,
        )
    
    def _parse_expression_or_assignment(self) -> Statement:
        """Parse an expression statement or assignment."""
        location = self._current.location
        expr = self._parse_expression()
        
        # Check for assignment
        if self._match(TokenType.ASSIGN):
            value = self._parse_expression()
            self._match(TokenType.NEWLINE)
            return AssignmentStmt(location=location, target=expr, value=value)
        
        self._match(TokenType.NEWLINE)
        return ExpressionStmt(location=location, expression=expr)
    
    # =========================================================================
    # Expression Parsing (Precedence Climbing)
    # =========================================================================
    
    def _parse_expression(self) -> Expression:
        """Parse an expression using precedence climbing."""
        return self._parse_precedence(Precedence.OR)
    
    def _parse_precedence(self, min_precedence: Precedence) -> Expression:
        """Parse expression with given minimum precedence."""
        left = self._parse_unary()
        
        while True:
            op_type = self._current.type
            if op_type not in OP_PRECEDENCE:
                break
            
            precedence = OP_PRECEDENCE[op_type]
            if precedence < min_precedence:
                break
            
            op_token = self._advance()
            
            # Right-associative for power
            if op_type == TokenType.DOUBLE_STAR:
                right = self._parse_precedence(precedence)
            else:
                right = self._parse_precedence(Precedence(precedence + 1))
            
            left = BinaryOp(
                location=left.location,
                left=left,
                operator=BINARY_OPS[op_type],
                right=right,
            )
        
        return left
    
    def _parse_unary(self) -> Expression:
        """Parse unary expression."""
        if self._check(TokenType.MINUS):
            location = self._current.location
            self._advance()
            operand = self._parse_unary()
            return UnaryOp(location=location, operator=UnaryOperator.NEG, operand=operand)
        elif self._check(TokenType.PLUS):
            location = self._current.location
            self._advance()
            operand = self._parse_unary()
            return UnaryOp(location=location, operator=UnaryOperator.POS, operand=operand)
        elif self._check(TokenType.NOT):
            location = self._current.location
            self._advance()
            operand = self._parse_unary()
            return UnaryOp(location=location, operator=UnaryOperator.NOT, operand=operand)
        
        return self._parse_call()
    
    def _parse_call(self) -> Expression:
        """Parse call expressions, indexing, and attribute access."""
        expr = self._parse_primary()
        
        while True:
            if self._match(TokenType.LPAREN):
                # Function call
                args: list[Expression] = []
                kwargs: list[tuple[str, Expression]] = []
                
                if not self._check(TokenType.RPAREN):
                    self._parse_arguments(args, kwargs)
                
                self._expect(TokenType.RPAREN, "Expected ')' after arguments")
                
                expr = CallExpr(
                    location=expr.location,
                    callee=expr,
                    arguments=tuple(args),
                    keyword_args=tuple(kwargs),
                )
            elif self._match(TokenType.LBRACKET):
                # Index access
                index = self._parse_expression()
                self._expect(TokenType.RBRACKET, "Expected ']' after index")
                expr = IndexExpr(location=expr.location, object=expr, index=index)
            elif self._match(TokenType.DOT):
                # Attribute access
                attr_token = self._expect(TokenType.IDENTIFIER, "Expected attribute name")
                expr = AttributeExpr(location=expr.location, object=expr, attribute=attr_token.value)
            else:
                break
        
        return expr
    
    def _parse_arguments(self, args: list[Expression], kwargs: list[tuple[str, Expression]]) -> None:
        """Parse function call arguments."""
        while True:
            # Check for keyword argument
            if (self._check(TokenType.IDENTIFIER) and 
                self._peek().type in (TokenType.COLON, TokenType.ASSIGN)):
                name_token = self._advance()
                self._advance()  # consume : or =
                value = self._parse_expression()
                kwargs.append((name_token.value, value))
            else:
                args.append(self._parse_expression())
            
            if not self._match(TokenType.COMMA):
                break
    
    def _parse_primary(self) -> Expression:
        """Parse primary expressions (literals, identifiers, grouping)."""
        location = self._current.location
        
        # Literals
        if self._check(TokenType.INTEGER):
            token = self._advance()
            return Literal(location=location, value=token.value)
        elif self._check(TokenType.FLOAT):
            token = self._advance()
            return Literal(location=location, value=token.value)
        elif self._check(TokenType.STRING):
            token = self._advance()
            return Literal(location=location, value=token.value)
        elif self._check(TokenType.TRIPLE_STRING):
            token = self._advance()
            # Check if we're in natural language context
            if self._in_natural_language:
                self._in_natural_language = False
                return NaturalLanguageExpr(location=location, content=token.value)
            return Literal(location=location, value=token.value)
        elif self._check(TokenType.TRUE):
            self._advance()
            return Literal(location=location, value=True)
        elif self._check(TokenType.FALSE):
            self._advance()
            return Literal(location=location, value=False)
        elif self._check(TokenType.NONE):
            self._advance()
            return Literal(location=location, value=None)
        
        # Identifier
        elif self._check(TokenType.IDENTIFIER):
            token = self._advance()
            return Identifier(location=location, name=token.value)
        
        # Grouped expression
        elif self._match(TokenType.LPAREN):
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected ')' after expression")
            return expr
        
        # Tensor/list literal
        elif self._match(TokenType.LBRACKET):
            elements: list[Expression] = []
            if not self._check(TokenType.RBRACKET):
                while True:
                    elements.append(self._parse_expression())
                    if not self._match(TokenType.COMMA):
                        break
            self._expect(TokenType.RBRACKET, "Expected ']' after elements")
            return TensorExpr(location=location, elements=tuple(elements))
        
        raise self._error(f"Unexpected token in expression: {self._current.type.name}")


def parse(source: str, filename: str = "<stdin>") -> Program:
    """
    Convenience function to parse source code into an AST.
    
    Args:
        source: The source code to parse
        filename: Optional filename for error reporting
    
    Returns:
        The parsed Program AST
    """
    from trident.lexer import Lexer
    
    lexer = Lexer(source, filename)
    tokens = list(lexer.tokenize())
    parser = Parser(tokens, filename)
    return parser.parse()
