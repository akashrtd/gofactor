"""
Trident Runtime Executor.

Provides the runtime environment for executing compiled Trident code,
including model loading and execution context management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
from pathlib import Path
import importlib.util
import sys

from trident.runtime.device import DeviceManager, DeviceType


@dataclass
class ModelHandle:
    """Handle to a loaded AI model."""
    name: str
    category: str
    model: Any
    tokenizer: Any = None
    processor: Any = None
    
    def __call__(self, *args, **kwargs) -> Any:
        """Call the model with inputs."""
        return self.model(*args, **kwargs)


@dataclass
class TridentRuntime:
    """
    Runtime environment for Trident programs.
    
    Manages:
    - Device selection (TPU/GPU/CPU)
    - Model loading and caching
    - Execution context
    """
    
    device_manager: DeviceManager = field(default_factory=DeviceManager)
    _models: Dict[str, ModelHandle] = field(default_factory=dict)
    _context: Dict[str, Any] = field(default_factory=dict)
    _jax_available: bool = field(default=False, init=False)
    
    def __post_init__(self) -> None:
        """Initialize the runtime."""
        try:
            import jax
            import jax.numpy as jnp
            self._jax_available = True
        except ImportError:
            self._jax_available = False
    
    def load_model(self, model_name: str, category: str = "general") -> ModelHandle:
        """
        Load an AI model by name.
        
        Args:
            model_name: Name or path of the model to load
            category: Model category (ocr, nlp, vision, etc.)
        
        Returns:
            ModelHandle for the loaded model
        """
        cache_key = f"{category}:{model_name}"
        
        if cache_key in self._models:
            return self._models[cache_key]
        
        handle = self._load_model_impl(model_name, category)
        self._models[cache_key] = handle
        return handle
    
    def _load_model_impl(self, model_name: str, category: str) -> ModelHandle:
        """Implementation of model loading."""
        if category == "ocr":
            return self._load_ocr_model(model_name)
        elif category == "nlp":
            return self._load_nlp_model(model_name)
        elif category == "vision":
            return self._load_vision_model(model_name)
        else:
            return self._load_generic_model(model_name, category)
    
    def _load_ocr_model(self, model_name: str) -> ModelHandle:
        """Load an OCR model."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            # Map friendly names to HuggingFace model IDs
            model_mapping = {
                "qwen2.5-vl": "Qwen/Qwen2-VL-2B-Instruct",
                "qwen-vl": "Qwen/Qwen2-VL-2B-Instruct",
                "minicpm-o": "openbmb/MiniCPM-V-2_6",
                "got-ocr": "stepfun-ai/GOT-OCR2_0",
            }
            
            hf_model_id = model_mapping.get(model_name.lower(), model_name)
            
            processor = AutoProcessor.from_pretrained(hf_model_id)
            model = AutoModelForVision2Seq.from_pretrained(hf_model_id)
            
            return ModelHandle(
                name=model_name,
                category="ocr",
                model=model,
                processor=processor,
            )
        except Exception as e:
            # Return a placeholder if model loading fails
            return self._create_placeholder_model(model_name, "ocr", str(e))
    
    def _load_nlp_model(self, model_name: str) -> ModelHandle:
        """Load an NLP/LLM model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Map friendly names to HuggingFace model IDs
            model_mapping = {
                "gemma-3": "google/gemma-2-2b-it",
                "gemma": "google/gemma-2-2b-it",
                "qwen": "Qwen/Qwen2-0.5B-Instruct",
                "qwen-3": "Qwen/Qwen2-0.5B-Instruct",
                "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            }
            
            hf_model_id = model_mapping.get(model_name.lower(), model_name)
            
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            model = AutoModelForCausalLM.from_pretrained(hf_model_id)
            
            return ModelHandle(
                name=model_name,
                category="nlp",
                model=model,
                tokenizer=tokenizer,
            )
        except Exception as e:
            return self._create_placeholder_model(model_name, "nlp", str(e))
    
    def _load_vision_model(self, model_name: str) -> ModelHandle:
        """Load a vision model."""
        try:
            from transformers import AutoProcessor, AutoModel
            
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            return ModelHandle(
                name=model_name,
                category="vision",
                model=model,
                processor=processor,
            )
        except Exception as e:
            return self._create_placeholder_model(model_name, "vision", str(e))
    
    def _load_generic_model(self, model_name: str, category: str) -> ModelHandle:
        """Load a generic model."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            return ModelHandle(
                name=model_name,
                category=category,
                model=model,
                tokenizer=tokenizer,
            )
        except Exception as e:
            return self._create_placeholder_model(model_name, category, str(e))
    
    def _create_placeholder_model(self, model_name: str, category: str, error: str) -> ModelHandle:
        """Create a placeholder model that logs warnings."""
        class PlaceholderModel:
            def __init__(self, name: str, error: str):
                self.name = name
                self.error = error
            
            def __call__(self, *args, **kwargs):
                print(f"[Trident Warning] Model '{self.name}' not loaded: {self.error}")
                print(f"[Trident Warning] Running in placeholder mode - returning mock output")
                return {"text": f"[Placeholder output for {self.name}]"}
        
        return ModelHandle(
            name=model_name,
            category=category,
            model=PlaceholderModel(model_name, error),
        )
    
    def execute_file(self, filepath: str) -> Any:
        """
        Execute a compiled Trident file.
        
        Args:
            filepath: Path to the compiled Python file
        
        Returns:
            Result of execution
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load as Python module
        spec = importlib.util.spec_from_file_location("trident_module", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load: {filepath}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["trident_module"] = module
        
        # Inject runtime
        module.__dict__["__trident_runtime__"] = self
        
        spec.loader.exec_module(module)
        return module
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self._context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self._context.get(key, default)
    
    def info(self) -> str:
        """Get runtime information."""
        lines = [
            "Trident Runtime",
            "=" * 40,
            f"JAX Available: {self._jax_available}",
            "",
            self.device_manager.info(),
            "",
            f"Loaded Models: {len(self._models)}",
        ]
        
        for key, handle in self._models.items():
            lines.append(f"  - {key}")
        
        return "\n".join(lines)
