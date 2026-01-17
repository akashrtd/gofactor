"""
Manifest file handling for Trident packages.

The manifest file (trident.toml) defines:
- Package metadata (name, version, description)
- Dependencies
- Build configuration
- Model requirements
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class Dependency:
    """A package dependency."""
    name: str
    version: str = "*"
    optional: bool = False
    features: List[str] = field(default_factory=list)
    
    @classmethod
    def from_string(cls, spec: str) -> "Dependency":
        """Parse a dependency string like 'package>=1.0.0'."""
        # Simple parsing
        if ">=" in spec:
            name, version = spec.split(">=", 1)
            return cls(name=name.strip(), version=f">={version.strip()}")
        elif "==" in spec:
            name, version = spec.split("==", 1)
            return cls(name=name.strip(), version=version.strip())
        else:
            return cls(name=spec.strip())
    
    def __str__(self) -> str:
        if self.version == "*":
            return self.name
        return f"{self.name}{self.version}"


@dataclass
class ModelRequirement:
    """A required AI model."""
    name: str
    category: str  # ocr, nlp, vision
    version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name, "category": self.category}
        if self.version:
            d["version"] = self.version
        return d


@dataclass
class Manifest:
    """
    Trident package manifest (trident.toml).
    """
    name: str
    version: str = "0.1.0"
    description: str = ""
    authors: List[str] = field(default_factory=list)
    license: str = "MIT"
    
    # Entry points
    main: Optional[str] = None
    
    # Dependencies
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    
    # Model requirements
    models: List[ModelRequirement] = field(default_factory=list)
    
    # Hardware preferences
    hardware: Dict[str, Any] = field(default_factory=dict)
    
    # Build settings
    build: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, path: Path) -> "Manifest":
        """Load manifest from trident.toml file."""
        try:
            import tomli
        except ImportError:
            import tomllib as tomli
        
        with open(path, "rb") as f:
            data = tomli.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        """Create manifest from dictionary."""
        project = data.get("project", data)
        
        models = []
        for model_data in data.get("models", []):
            models.append(ModelRequirement(
                name=model_data["name"],
                category=model_data.get("category", "general"),
                version=model_data.get("version"),
            ))
        
        return cls(
            name=project.get("name", "unnamed"),
            version=project.get("version", "0.1.0"),
            description=project.get("description", ""),
            authors=project.get("authors", []),
            license=project.get("license", "MIT"),
            main=project.get("main"),
            dependencies=data.get("dependencies", {}),
            dev_dependencies=data.get("dev-dependencies", {}),
            models=models,
            hardware=data.get("hardware", {}),
            build=data.get("build", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "project": {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "authors": self.authors,
                "license": self.license,
                "main": self.main,
            },
            "dependencies": self.dependencies,
            "dev-dependencies": self.dev_dependencies,
            "models": [m.to_dict() for m in self.models],
            "hardware": self.hardware,
            "build": self.build,
        }
    
    def to_toml(self) -> str:
        """Convert manifest to TOML string."""
        lines = [
            "[project]",
            f'name = "{self.name}"',
            f'version = "{self.version}"',
            f'description = "{self.description}"',
        ]
        
        if self.authors:
            authors_str = ", ".join(f'"{a}"' for a in self.authors)
            lines.append(f"authors = [{authors_str}]")
        
        lines.append(f'license = "{self.license}"')
        
        if self.main:
            lines.append(f'main = "{self.main}"')
        
        lines.append("")
        lines.append("[dependencies]")
        for name, version in self.dependencies.items():
            lines.append(f'{name} = "{version}"')
        
        if self.dev_dependencies:
            lines.append("")
            lines.append("[dev-dependencies]")
            for name, version in self.dev_dependencies.items():
                lines.append(f'{name} = "{version}"')
        
        if self.models:
            lines.append("")
            lines.append("[[models]]")
            for model in self.models:
                lines.append(f'name = "{model.name}"')
                lines.append(f'category = "{model.category}"')
                if model.version:
                    lines.append(f'version = "{model.version}"')
        
        if self.hardware:
            lines.append("")
            lines.append("[hardware]")
            for key, value in self.hardware.items():
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                else:
                    lines.append(f"{key} = {value}")
        
        return "\n".join(lines)
    
    def save(self, path: Path) -> None:
        """Save manifest to file."""
        path.write_text(self.to_toml())
    
    def get_dependencies(self) -> List[Dependency]:
        """Get all dependencies as Dependency objects."""
        deps = []
        for name, version in self.dependencies.items():
            deps.append(Dependency(name=name, version=version))
        return deps


def create_default_manifest(name: str, path: Path) -> Manifest:
    """Create a default manifest for a new project."""
    manifest = Manifest(
        name=name,
        version="0.1.0",
        description=f"A Trident project: {name}",
        main="src/main.tri",
    )
    manifest.save(path / "trident.toml")
    return manifest
