"""
Dependency Resolver for Trident packages.

Implements a SAT-based dependency resolution algorithm to find
compatible versions of all required packages.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto

from trident.librarian.manifest import Dependency


class VersionOperator(Enum):
    """Version comparison operators."""
    EQ = auto()   # ==
    NE = auto()   # !=
    LT = auto()   # <
    LE = auto()   # <=
    GT = auto()   # >
    GE = auto()   # >=
    ANY = auto()  # *


@dataclass
class Version:
    """Semantic version representation."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    
    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse a version string like '1.2.3' or '1.2.3-beta'."""
        # Remove leading 'v' if present
        if version_str.startswith("v"):
            version_str = version_str[1:]
        
        # Handle prerelease
        prerelease = None
        if "-" in version_str:
            version_str, prerelease = version_str.split("-", 1)
        
        parts = version_str.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        
        return cls(major, minor, patch, prerelease)
    
    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += f"-{self.prerelease}"
        return base
    
    def __lt__(self, other: "Version") -> bool:
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        # Prerelease versions are less than release versions
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        return (self.prerelease or "") < (other.prerelease or "")
    
    def __le__(self, other: "Version") -> bool:
        return self == other or self < other
    
    def __gt__(self, other: "Version") -> bool:
        return not self <= other
    
    def __ge__(self, other: "Version") -> bool:
        return not self < other
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return False
        return (self.major, self.minor, self.patch, self.prerelease) == \
               (other.major, other.minor, other.patch, other.prerelease)
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))


@dataclass
class VersionConstraint:
    """A constraint on package versions."""
    operator: VersionOperator
    version: Optional[Version] = None
    
    @classmethod
    def parse(cls, spec: str) -> "VersionConstraint":
        """Parse a version constraint like '>=1.0.0' or '*'."""
        spec = spec.strip()
        
        if spec == "*" or not spec:
            return cls(operator=VersionOperator.ANY)
        
        if spec.startswith(">="):
            return cls(VersionOperator.GE, Version.parse(spec[2:]))
        elif spec.startswith("<="):
            return cls(VersionOperator.LE, Version.parse(spec[2:]))
        elif spec.startswith("=="):
            return cls(VersionOperator.EQ, Version.parse(spec[2:]))
        elif spec.startswith("!="):
            return cls(VersionOperator.NE, Version.parse(spec[2:]))
        elif spec.startswith(">"):
            return cls(VersionOperator.GT, Version.parse(spec[1:]))
        elif spec.startswith("<"):
            return cls(VersionOperator.LT, Version.parse(spec[1:]))
        else:
            # Exact version
            return cls(VersionOperator.EQ, Version.parse(spec))
    
    def matches(self, version: Version) -> bool:
        """Check if a version satisfies this constraint."""
        if self.operator == VersionOperator.ANY:
            return True
        
        if self.version is None:
            return True
        
        ops = {
            VersionOperator.EQ: lambda v: v == self.version,
            VersionOperator.NE: lambda v: v != self.version,
            VersionOperator.LT: lambda v: v < self.version,
            VersionOperator.LE: lambda v: v <= self.version,
            VersionOperator.GT: lambda v: v > self.version,
            VersionOperator.GE: lambda v: v >= self.version,
        }
        
        return ops[self.operator](version)


@dataclass
class ResolvedPackage:
    """A resolved package with selected version."""
    name: str
    version: Version
    dependencies: List[Dependency] = field(default_factory=list)


class ResolutionError(Exception):
    """Error during dependency resolution."""
    pass


@dataclass
class DependencyResolver:
    """
    Resolves package dependencies.
    
    Uses a simple backtracking algorithm to find compatible versions.
    For production use, this should be replaced with a SAT solver.
    """
    
    # Mock registry (in production, this would query a real registry)
    _registry: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize with some mock packages."""
        self._registry = {
            "vision-utils": {
                "1.0.0": {"dependencies": {}},
                "1.1.0": {"dependencies": {}},
            },
            "nlp-helpers": {
                "2.0.0": {"dependencies": {}},
                "2.1.0": {"dependencies": {"tokenizers": ">=0.1.0"}},
            },
            "tokenizers": {
                "0.1.0": {"dependencies": {}},
                "0.2.0": {"dependencies": {}},
            },
            "tensor-ops": {
                "1.0.0": {"dependencies": {}},
            },
        }
    
    def available_versions(self, package: str) -> List[Version]:
        """Get available versions for a package."""
        if package not in self._registry:
            return []
        return sorted([Version.parse(v) for v in self._registry[package].keys()], reverse=True)
    
    def get_dependencies(self, package: str, version: Version) -> List[Dependency]:
        """Get dependencies for a specific package version."""
        if package not in self._registry:
            return []
        
        version_str = str(version)
        if version_str not in self._registry[package]:
            return []
        
        deps = self._registry[package][version_str].get("dependencies", {})
        return [Dependency(name=n, version=v) for n, v in deps.items()]
    
    def resolve(self, requirements: List[Dependency]) -> List[ResolvedPackage]:
        """
        Resolve dependencies and return a list of packages to install.
        
        Args:
            requirements: List of top-level dependencies
        
        Returns:
            List of resolved packages with versions
        
        Raises:
            ResolutionError: If no valid resolution exists
        """
        resolved: Dict[str, ResolvedPackage] = {}
        constraints: Dict[str, List[VersionConstraint]] = {}
        
        # Collect all requirements
        to_process = list(requirements)
        
        while to_process:
            dep = to_process.pop(0)
            
            # Add constraint
            if dep.name not in constraints:
                constraints[dep.name] = []
            constraints[dep.name].append(VersionConstraint.parse(dep.version))
            
            # Skip if already resolved
            if dep.name in resolved:
                # Verify constraint is still satisfied
                if not all(c.matches(resolved[dep.name].version) for c in constraints[dep.name]):
                    raise ResolutionError(f"Conflicting requirements for {dep.name}")
                continue
            
            # Find compatible version
            available = self.available_versions(dep.name)
            
            if not available:
                # Package not in registry - assume it's available externally
                resolved[dep.name] = ResolvedPackage(
                    name=dep.name,
                    version=Version.parse("0.0.0"),
                )
                continue
            
            # Find first version that satisfies all constraints
            found = False
            for version in available:
                if all(c.matches(version) for c in constraints[dep.name]):
                    # Get package dependencies
                    pkg_deps = self.get_dependencies(dep.name, version)
                    
                    resolved[dep.name] = ResolvedPackage(
                        name=dep.name,
                        version=version,
                        dependencies=pkg_deps,
                    )
                    
                    # Add transitive dependencies
                    to_process.extend(pkg_deps)
                    found = True
                    break
            
            if not found:
                raise ResolutionError(
                    f"No compatible version found for {dep.name} "
                    f"(constraints: {[str(c) for c in constraints[dep.name]]})"
                )
        
        return list(resolved.values())
    
    def generate_lockfile(self, resolved: List[ResolvedPackage]) -> str:
        """Generate a lockfile from resolved packages."""
        lines = ["# Trident Lockfile", "# Generated by librarian", ""]
        
        for pkg in sorted(resolved, key=lambda p: p.name):
            lines.append(f"[[package]]")
            lines.append(f'name = "{pkg.name}"')
            lines.append(f'version = "{pkg.version}"')
            if pkg.dependencies:
                deps = ", ".join(f'"{d.name}"' for d in pkg.dependencies)
                lines.append(f"dependencies = [{deps}]")
            lines.append("")
        
        return "\n".join(lines)
