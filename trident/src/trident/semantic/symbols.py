"""
Symbol Table for the Trident programming language.

Manages scopes and symbol resolution during semantic analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, Iterator
from enum import Enum, auto

from trident.semantic.types import Type, ANY, UNKNOWN
from trident.lexer.tokens import SourceLocation


class SymbolKind(Enum):
    """Classification of symbols."""
    VARIABLE = auto()
    CONSTANT = auto()
    FUNCTION = auto()
    PIPELINE = auto()
    PARAMETER = auto()
    MODEL = auto()
    TYPE = auto()
    MODULE = auto()
    STRUCT = auto()


@dataclass
class Symbol:
    """
    Represents a named entity in the program.
    
    Attributes:
        name: The symbol's name
        kind: What kind of symbol this is
        type: The symbol's type
        location: Where the symbol was defined
        is_mutable: Whether the symbol can be reassigned
        is_builtin: Whether this is a built-in symbol
    """
    name: str
    kind: SymbolKind
    type: Type
    location: Optional[SourceLocation] = None
    is_mutable: bool = True
    is_builtin: bool = False
    
    def __str__(self) -> str:
        return f"Symbol({self.name}: {self.type})"


@dataclass
class Scope:
    """
    A scope containing symbol bindings.
    
    Scopes are organized in a tree structure with parent-child relationships.
    Symbol lookup walks up the scope chain until a match is found.
    """
    name: str
    parent: Optional["Scope"] = None
    symbols: dict[str, Symbol] = field(default_factory=dict)
    children: list["Scope"] = field(default_factory=list)
    
    def define(self, symbol: Symbol) -> bool:
        """
        Define a symbol in this scope.
        
        Returns:
            True if defined successfully, False if already exists
        """
        if symbol.name in self.symbols:
            return False
        self.symbols[symbol.name] = symbol
        return True
    
    def lookup(self, name: str, local_only: bool = False) -> Optional[Symbol]:
        """
        Look up a symbol by name.
        
        Args:
            name: The symbol name to look up
            local_only: If True, only search this scope (not parents)
        
        Returns:
            The symbol if found, None otherwise
        """
        if name in self.symbols:
            return self.symbols[name]
        if not local_only and self.parent:
            return self.parent.lookup(name)
        return None
    
    def is_defined_locally(self, name: str) -> bool:
        """Check if a symbol is defined in this exact scope."""
        return name in self.symbols
    
    def create_child(self, name: str) -> "Scope":
        """Create a child scope."""
        child = Scope(name=name, parent=self)
        self.children.append(child)
        return child
    
    def all_symbols(self) -> Iterator[Symbol]:
        """Iterate over all symbols in this scope (not parents)."""
        yield from self.symbols.values()
    
    def depth(self) -> int:
        """Get the nesting depth of this scope."""
        if self.parent is None:
            return 0
        return 1 + self.parent.depth()
    
    def __str__(self) -> str:
        return f"Scope({self.name}, {len(self.symbols)} symbols)"


@dataclass
class SymbolTable:
    """
    Manages all scopes during compilation.
    
    Provides convenience methods for scope management and symbol operations.
    """
    global_scope: Scope = field(default_factory=lambda: Scope("global"))
    current_scope: Scope = field(init=False)
    
    def __post_init__(self) -> None:
        self.current_scope = self.global_scope
        self._init_builtins()
    
    def _init_builtins(self) -> None:
        """Initialize built-in symbols."""
        from trident.semantic.types import (
            INT, FLOAT, STRING, BOOL, IMAGE, DOCUMENT, ANY, NONE,
            FunctionType, TensorType, DictType,
        )
        
        # Built-in types
        type_symbols = [
            ("Int", INT),
            ("Float", FLOAT),
            ("String", STRING),
            ("Bool", BOOL),
            ("Image", IMAGE),
            ("Document", DOCUMENT),
            ("Any", ANY),
            ("None", NONE),
            ("Tensor", TensorType()),
            # Lowercase aliases
            ("int", INT),
            ("float", FLOAT),
            ("string", STRING),
            ("bool", BOOL),
            ("image", IMAGE),
            ("document", DOCUMENT),
            ("tensor", TensorType()),
            ("json", DictType(STRING, ANY)), # Add json alias
        ]
        
        for name, typ in type_symbols:
            self.define(Symbol(
                name=name,
                kind=SymbolKind.TYPE,
                type=typ,
                is_mutable=False,
                is_builtin=True,
            ))
        
        # Built-in functions
        builtin_funcs = [
            ("print", FunctionType((ANY,), NONE)),
            ("len", FunctionType((ANY,), INT)),
            ("range", FunctionType((INT,), ANY)),  # Returns iterator
            ("str", FunctionType((ANY,), STRING)),
            ("int", FunctionType((ANY,), INT)),
            ("float", FunctionType((ANY,), FLOAT)),
            ("bool", FunctionType((ANY,), BOOL)),
            ("type", FunctionType((ANY,), STRING)),
            ("abs", FunctionType((FLOAT,), FLOAT)),
            ("min", FunctionType((ANY, ANY), ANY)),
            ("max", FunctionType((ANY, ANY), ANY)),
            ("sum", FunctionType((ANY,), FLOAT)),
            ("sqrt", FunctionType((FLOAT,), FLOAT)),
            ("softmax", FunctionType((TensorType(),), TensorType())),
            ("zeros", FunctionType((ANY,), TensorType())),
            ("ones", FunctionType((ANY,), TensorType())),
            ("randn", FunctionType((ANY,), TensorType())),
        ]
        
        for name, typ in builtin_funcs:
            self.define(Symbol(
                name=name,
                kind=SymbolKind.FUNCTION,
                type=typ,
                is_mutable=False,
                is_builtin=True,
            ))
            
        # Built-in modules
        builtin_modules = ["vision", "ocr", "nlp"]
        for name in builtin_modules:
            self.define(Symbol(
                name=name,
                kind=SymbolKind.MODULE,
                type=ANY,
                is_mutable=False,
                is_builtin=True,
            ))
    
    def enter_scope(self, name: str) -> Scope:
        """Enter a new child scope."""
        self.current_scope = self.current_scope.create_child(name)
        return self.current_scope
    
    def exit_scope(self) -> Scope:
        """Exit current scope and return to parent."""
        if self.current_scope.parent:
            old = self.current_scope
            self.current_scope = self.current_scope.parent
            return old
        return self.current_scope
    
    def define(self, symbol: Symbol) -> bool:
        """Define a symbol in the current scope."""
        return self.current_scope.define(symbol)
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol starting from current scope."""
        return self.current_scope.lookup(name)
    
    def lookup_local(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in current scope only."""
        return self.current_scope.lookup(name, local_only=True)
    
    def is_global_scope(self) -> bool:
        """Check if currently in global scope."""
        return self.current_scope is self.global_scope
    
    def scope_depth(self) -> int:
        """Get current scope depth."""
        return self.current_scope.depth()
