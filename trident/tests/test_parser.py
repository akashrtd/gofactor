"""
Tests for the Trident parser.
"""

import pytest
from trident.parser import parse, Parser, ParseError
from trident.parser.ast import (
    Program,
    FunctionDef,
    PipelineDef,
    LetStatement,
    ReturnStatement,
    IfStatement,
    ForStatement,
    ExpressionStmt,
    Literal,
    Identifier,
    BinaryOp,
    CallExpr,
    Annotation,
)
from trident.lexer import Lexer


class TestParser:
    """Test the parser's AST generation."""
    
    def test_empty_program(self):
        """Empty source produces empty program."""
        ast = parse("")
        assert isinstance(ast, Program)
        assert len(ast.statements) == 0
    
    def test_simple_function(self):
        """Parse a simple function definition."""
        source = """fn hello():
    pass"""
        
        ast = parse(source)
        assert len(ast.statements) == 1
        assert isinstance(ast.statements[0], FunctionDef)
        assert ast.statements[0].name == "hello"
    
    def test_function_with_params(self):
        """Parse function with parameters."""
        source = """fn add(a: Int, b: Int) -> Int:
    return a + b"""
        
        ast = parse(source)
        func = ast.statements[0]
        
        assert isinstance(func, FunctionDef)
        assert func.name == "add"
        assert len(func.params) == 2
        assert func.params[0].name == "a"
        assert func.params[1].name == "b"
    
    def test_let_statement(self):
        """Parse let statements."""
        source = """fn test():
    let x = 42
    let y: Float = 3.14"""
        
        ast = parse(source)
        func = ast.statements[0]
        
        assert len(func.body) == 2
        assert isinstance(func.body[0], LetStatement)
        assert func.body[0].name == "x"
        assert isinstance(func.body[0].value, Literal)
        assert func.body[0].value.value == 42
    
    def test_if_statement(self):
        """Parse if statements."""
        source = """fn test():
    if x > 0:
        return 1
    else:
        return 0"""
        
        ast = parse(source)
        func = ast.statements[0]
        
        assert len(func.body) == 1
        assert isinstance(func.body[0], IfStatement)
        assert len(func.body[0].then_body) == 1
        assert len(func.body[0].else_body) == 1
    
    def test_for_loop(self):
        """Parse for loops."""
        source = """fn test():
    for i in range(10):
        print(i)"""
        
        ast = parse(source)
        func = ast.statements[0]
        
        assert len(func.body) == 1
        assert isinstance(func.body[0], ForStatement)
        assert func.body[0].variable == "i"
    
    def test_binary_operations(self):
        """Parse binary operations."""
        source = """fn test():
    let x = 1 + 2 * 3"""
        
        ast = parse(source)
        func = ast.statements[0]
        let_stmt = func.body[0]
        
        assert isinstance(let_stmt.value, BinaryOp)
    
    def test_function_call(self):
        """Parse function calls."""
        source = """fn test():
    print("hello", x: 42)"""
        
        ast = parse(source)
        func = ast.statements[0]
        expr_stmt = func.body[0]
        
        assert isinstance(expr_stmt, ExpressionStmt)
        assert isinstance(expr_stmt.expression, CallExpr)
        
        call = expr_stmt.expression
        assert len(call.arguments) == 1
        assert len(call.keyword_args) == 1
    
    def test_pipeline_definition(self):
        """Parse pipeline definitions."""
        source = """pipeline MyPipeline:
    fn process():
        pass"""
        
        ast = parse(source)
        
        assert len(ast.statements) == 1
        assert isinstance(ast.statements[0], PipelineDef)
        assert ast.statements[0].name == "MyPipeline"
    
    def test_annotations(self):
        """Parse annotations on definitions."""
        source = """@intent("Process documents")
@model(ocr: "qwen")
fn process():
    pass"""
        
        ast = parse(source)
        func = ast.statements[0]
        
        assert isinstance(func, FunctionDef)
        assert len(func.annotations) == 2
        assert func.annotations[0].name == "intent"
        assert func.annotations[1].name == "model"
    
    def test_tensor_literal(self):
        """Parse tensor/list literals."""
        source = """fn test():
    let arr = [1, 2, 3, 4]"""
        
        ast = parse(source)
        func = ast.statements[0]
        let_stmt = func.body[0]
        
        from trident.parser.ast import TensorExpr
        assert isinstance(let_stmt.value, TensorExpr)
        assert len(let_stmt.value.elements) == 4
    
    def test_attribute_access(self):
        """Parse attribute access."""
        source = """fn test():
    let y = x.shape"""
        
        ast = parse(source)
        func = ast.statements[0]
        let_stmt = func.body[0]
        
        from trident.parser.ast import AttributeExpr
        assert isinstance(let_stmt.value, AttributeExpr)
        assert let_stmt.value.attribute == "shape"
    
    def test_index_access(self):
        """Parse index/subscript access."""
        source = """fn test():
    let y = arr[0]"""
        
        ast = parse(source)
        func = ast.statements[0]
        let_stmt = func.body[0]
        
        from trident.parser.ast import IndexExpr
        assert isinstance(let_stmt.value, IndexExpr)


class TestParserEdgeCases:
    """Test parser edge cases and error handling."""
    
    def test_nested_functions(self):
        """Parse nested function calls."""
        source = """fn test():
    let x = foo(bar(baz(1)))"""
        
        ast = parse(source)
        # Should parse without error
        assert isinstance(ast, Program)
    
    def test_complex_expression(self):
        """Parse complex mathematical expressions."""
        source = """fn test():
    let x = (a + b) * (c - d) / e ** 2"""
        
        ast = parse(source)
        assert isinstance(ast, Program)
    
    def test_multiline_string_in_call(self):
        """Parse multiline strings in function calls."""
        source = '''fn test():
    let result = llm.query(context, """
        Extract the data.
        Return as JSON.
    """)'''
        
        ast = parse(source)
        assert isinstance(ast, Program)
