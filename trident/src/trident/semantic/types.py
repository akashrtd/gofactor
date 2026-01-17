"""
Type System for the Trident programming language.

The type system is designed to be:
1. Gradual - Allow mixing typed and untyped code
2. Tensor-aware - First-class support for shaped tensors
3. AI-friendly - Model and Pipeline types for AI primitives

Type Hierarchy:
    Type (abstract)
    ├── PrimitiveType (Int, Float, String, Bool)
    ├── TensorType (shaped arrays)
    ├── FunctionType (callable types)
    ├── PipelineType (AI pipeline types)
    ├── ModelType (AI model references)
    ├── ContainerType
    │   ├── ListType
    │   └── DictType
    ├── AnyType (top type)
    ├── NoneType (unit type)
    └── UnknownType (type inference placeholder)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence
from enum import Enum, auto


class TypeKind(Enum):
    """Classification of types for quick checks."""
    PRIMITIVE = auto()
    TENSOR = auto()
    FUNCTION = auto()
    PIPELINE = auto()
    MODEL = auto()
    CONTAINER = auto()
    ANY = auto()
    NONE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class Type(ABC):
    """Base class for all types in Trident."""
    
    @property
    @abstractmethod
    def kind(self) -> TypeKind:
        """Get the kind of this type."""
        pass
    
    @abstractmethod
    def is_assignable_from(self, other: "Type") -> bool:
        """Check if a value of 'other' type can be assigned to this type."""
        pass
    
    def is_numeric(self) -> bool:
        """Check if this is a numeric type."""
        return False
    
    def is_tensor_like(self) -> bool:
        """Check if this type supports tensor operations."""
        return False
    
    def __str__(self) -> str:
        return self.__class__.__name__


# =============================================================================
# Primitive Types
# =============================================================================

@dataclass(frozen=True)
class PrimitiveType(Type):
    """Primitive types: Int, Float, String, Bool."""
    name: str
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.PRIMITIVE
    
    def is_assignable_from(self, other: Type) -> bool:
        if isinstance(other, AnyType):
            return True
        if isinstance(other, UnknownType):
            return True
        if not isinstance(other, PrimitiveType):
            return False
        # Allow numeric coercion: Int -> Float
        if self.name == "Float" and other.name == "Int":
            return True
        return self.name == other.name
    
    def is_numeric(self) -> bool:
        return self.name in ("Int", "Float")
    
    def __str__(self) -> str:
        return self.name


# Singleton primitive types
INT = PrimitiveType("Int")
FLOAT = PrimitiveType("Float")
STRING = PrimitiveType("String")
BOOL = PrimitiveType("Bool")


# =============================================================================
# Special Types
# =============================================================================

@dataclass(frozen=True)
class ImageType(Type):
    """Type for image data."""
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.PRIMITIVE
    
    def is_assignable_from(self, other: Type) -> bool:
        if isinstance(other, (AnyType, UnknownType)):
            return True
        return isinstance(other, ImageType)
    
    def __str__(self) -> str:
        return "Image"


@dataclass(frozen=True)
class DocumentType(Type):
    """Type for document data (OCR output, etc.)."""
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.PRIMITIVE
    
    def is_assignable_from(self, other: Type) -> bool:
        if isinstance(other, (AnyType, UnknownType)):
            return True
        return isinstance(other, DocumentType)
    
    def __str__(self) -> str:
        return "Document"


IMAGE = ImageType()
DOCUMENT = DocumentType()


@dataclass(frozen=True)
class AnyType(Type):
    """Top type - all types are assignable to Any."""
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.ANY
    
    def is_assignable_from(self, other: Type) -> bool:
        return True  # Any accepts everything
    
    def __str__(self) -> str:
        return "Any"


@dataclass(frozen=True)
class NoneType(Type):
    """Unit type for None/null values."""
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.NONE
    
    def is_assignable_from(self, other: Type) -> bool:
        if isinstance(other, (AnyType, UnknownType)):
            return True
        return isinstance(other, NoneType)
    
    def __str__(self) -> str:
        return "None"


@dataclass(frozen=True)
class UnknownType(Type):
    """Placeholder for type inference - will be resolved later."""
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.UNKNOWN
    
    def is_assignable_from(self, other: Type) -> bool:
        return True  # Unknown accepts anything during inference
    
    def __str__(self) -> str:
        return "?"


ANY = AnyType()
NONE = NoneType()
UNKNOWN = UnknownType()


# =============================================================================
# Tensor Types
# =============================================================================

@dataclass(frozen=True)
class TensorType(Type):
    """
    Tensor type with optional shape information.
    
    Examples:
        Tensor[Float]           - unshapen tensor of floats
        Tensor[Float, 32, 64]   - 2D tensor with shape (32, 64)
        Tensor[Float, *, 64]    - 2D tensor with dynamic first dim
    """
    dtype: Type = field(default_factory=lambda: FLOAT)
    shape: tuple[int | str, ...] = ()  # str for dynamic dims like "*"
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.TENSOR
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)
    
    def is_assignable_from(self, other: Type) -> bool:
        if isinstance(other, (AnyType, UnknownType)):
            return True
        if not isinstance(other, TensorType):
            return False
        
        # Check dtype compatibility
        if not self.dtype.is_assignable_from(other.dtype):
            return False
        
        # If we have no shape, accept any shape
        if not self.shape:
            return True
        
        # If other has no shape, accept
        if not other.shape:
            return True
        
        # Check shape compatibility
        if len(self.shape) != len(other.shape):
            return False
        
        for s1, s2 in zip(self.shape, other.shape):
            if s1 == "*" or s2 == "*":
                continue  # Dynamic dims are compatible
            if s1 != s2:
                return False
        
        return True
    
    def is_numeric(self) -> bool:
        return self.dtype.is_numeric()
    
    def is_tensor_like(self) -> bool:
        return True
    
    def __str__(self) -> str:
        if not self.shape:
            return f"Tensor[{self.dtype}]"
        shape_str = ", ".join(str(s) for s in self.shape)
        return f"Tensor[{self.dtype}, {shape_str}]"


# =============================================================================
# Function Types
# =============================================================================

@dataclass(frozen=True)
class FunctionType(Type):
    """
    Function type with parameter and return types.
    
    Example: (Int, Int) -> Int
    """
    param_types: tuple[Type, ...]
    return_type: Type
    param_names: tuple[str, ...] = ()  # Optional parameter names
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.FUNCTION
    
    def is_assignable_from(self, other: Type) -> bool:
        if isinstance(other, (AnyType, UnknownType)):
            return True
        if not isinstance(other, FunctionType):
            return False
        
        # Check parameter types (contravariant)
        if len(self.param_types) != len(other.param_types):
            return False
        
        for p1, p2 in zip(self.param_types, other.param_types):
            if not p2.is_assignable_from(p1):  # Contravariant
                return False
        
        # Check return type (covariant)
        return self.return_type.is_assignable_from(other.return_type)
    
    def __str__(self) -> str:
        params = ", ".join(str(p) for p in self.param_types)
        return f"({params}) -> {self.return_type}"


# =============================================================================
# AI-Specific Types
# =============================================================================

@dataclass(frozen=True)
class ModelType(Type):
    """
    Type for AI model references.
    
    Represents models like OCR, NLP, vision models with their
    expected input and output types.
    """
    name: str
    input_type: Type = field(default_factory=lambda: ANY)
    output_type: Type = field(default_factory=lambda: ANY)
    category: str = "general"  # ocr, nlp, vision, etc.
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.MODEL
    
    def is_assignable_from(self, other: Type) -> bool:
        if isinstance(other, (AnyType, UnknownType)):
            return True
        if not isinstance(other, ModelType):
            return False
        return self.name == other.name
    
    def __str__(self) -> str:
        return f"Model[{self.name}]"


@dataclass(frozen=True)
class PipelineType(Type):
    """
    Type for AI pipeline definitions.
    
    Pipelines are compositions of models and transformations.
    """
    name: str
    stages: tuple[Type, ...] = ()
    input_type: Type = field(default_factory=lambda: ANY)
    output_type: Type = field(default_factory=lambda: ANY)
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.PIPELINE
    
    def is_assignable_from(self, other: Type) -> bool:
        if isinstance(other, (AnyType, UnknownType)):
            return True
        if not isinstance(other, PipelineType):
            return False
        return self.name == other.name
    
    def __str__(self) -> str:
        return f"Pipeline[{self.name}]"


# =============================================================================
# Container Types
# =============================================================================

@dataclass(frozen=True)
class ListType(Type):
    """List type with element type."""
    element_type: Type = field(default_factory=lambda: ANY)
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.CONTAINER
    
    def is_assignable_from(self, other: Type) -> bool:
        if isinstance(other, (AnyType, UnknownType)):
            return True
        if not isinstance(other, ListType):
            return False
        return self.element_type.is_assignable_from(other.element_type)
    
    def __str__(self) -> str:
        return f"List[{self.element_type}]"


@dataclass(frozen=True)
class DictType(Type):
    """Dictionary type with key and value types."""
    key_type: Type = field(default_factory=lambda: STRING)
    value_type: Type = field(default_factory=lambda: ANY)
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.CONTAINER
    
    def is_assignable_from(self, other: Type) -> bool:
        if isinstance(other, (AnyType, UnknownType)):
            return True
        if not isinstance(other, DictType):
            return False
        return (self.key_type.is_assignable_from(other.key_type) and
                self.value_type.is_assignable_from(other.value_type))
    
    def __str__(self) -> str:
        return f"Dict[{self.key_type}, {self.value_type}]"


# =============================================================================
# Type Utilities
# =============================================================================

def type_from_string(name: str) -> Type:
    """Convert a type name string to a Type object."""
    type_map = {
        "Int": INT,
        "Float": FLOAT,
        "String": STRING,
        "Bool": BOOL,
        "Image": IMAGE,
        "Document": DOCUMENT,
        "Any": ANY,
        "None": NONE,
        "Tensor": TensorType(),
    }
    return type_map.get(name, UNKNOWN)


def common_type(a: Type, b: Type) -> Type:
    """Find the common supertype of two types."""
    if a.is_assignable_from(b):
        return a
    if b.is_assignable_from(a):
        return b
    
    # Numeric promotion
    if a == INT and b == FLOAT:
        return FLOAT
    if a == FLOAT and b == INT:
        return FLOAT
    
    # Tensor promotion
    if isinstance(a, TensorType) and isinstance(b, TensorType):
        dtype = common_type(a.dtype, b.dtype)
        return TensorType(dtype=dtype)
    
    return ANY


def is_subtype(sub: Type, sup: Type) -> bool:
    """Check if 'sub' is a subtype of 'sup'."""
    return sup.is_assignable_from(sub)
