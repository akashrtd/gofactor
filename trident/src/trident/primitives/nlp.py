"""
NLP and LLM primitives for Trident.

Provides built-in operations for:
- LLM text generation and queries
- Text embeddings
- Tokenization
- Information extraction
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json


# =============================================================================
# LLM Operations
# =============================================================================

_llm_model = None
_llm_tokenizer = None


def _get_llm_model():
    """Get or initialize the LLM model."""
    global _llm_model, _llm_tokenizer
    
    if _llm_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Use a small efficient model by default
            model_id = "Qwen/Qwen2-0.5B-Instruct"
            
            _llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
            _llm_model = AutoModelForCausalLM.from_pretrained(model_id)
        except Exception as e:
            print(f"[Trident Warning] Could not load LLM: {e}")
            return None, None
    
    return _llm_model, _llm_tokenizer


def llm_query(
    context: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> str:
    """
    Query an LLM with context and prompt.
    
    Args:
        context: Context/document to reason about
        prompt: The question or instruction
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0-1)
        model: Optional specific model to use
    
    Returns:
        Generated text response
    """
    llm, tokenizer = _get_llm_model()
    
    if llm is None:
        return _mock_llm_response(context, prompt)
    
    try:
        # Build messages
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Answer based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion/Instruction:\n{prompt}"},
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        
        # Generate
        outputs = llm.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "assistant" in response.lower():
            parts = response.split("assistant")
            response = parts[-1].strip().lstrip(":").strip()
        
        return response
    except Exception as e:
        print(f"[Trident Warning] LLM query failed: {e}")
        return _mock_llm_response(context, prompt)


def _mock_llm_response(context: str, prompt: str) -> str:
    """Generate a mock response when LLM is not available."""
    # Try to extract structured data if the prompt looks like an extraction request
    if "extract" in prompt.lower():
        return json.dumps({
            "note": "Mock extraction - LLM not available",
            "context_length": len(context),
            "prompt": prompt[:100],
        }, indent=2)
    
    return f"[Mock response for: {prompt[:50]}...]"


def generate(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    stop: Optional[List[str]] = None,
) -> str:
    """
    Generate text continuation.
    
    Args:
        prompt: Starting text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stop: Stop sequences
    
    Returns:
        Generated continuation
    """
    llm, tokenizer = _get_llm_model()
    
    if llm is None:
        return "[LLM not available]"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        outputs = llm.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove original prompt from output
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        
        # Apply stop sequences
        if stop:
            for s in stop:
                if s in generated:
                    generated = generated[:generated.index(s)]
        
        return generated
    except Exception as e:
        return f"[Error: {e}]"


# =============================================================================
# Embeddings
# =============================================================================

_embedding_model = None


def _get_embedding_model():
    """Get or initialize the embedding model."""
    global _embedding_model
    
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                model_id = "sentence-transformers/all-MiniLM-L6-v2"
                _embedding_model = {
                    "tokenizer": AutoTokenizer.from_pretrained(model_id),
                    "model": AutoModel.from_pretrained(model_id),
                }
            except Exception as e:
                print(f"[Trident Warning] Could not load embedding model: {e}")
                return None
    
    return _embedding_model


def embed_text(text: Union[str, List[str]]) -> Any:
    """
    Generate embeddings for text.
    
    Args:
        text: Single string or list of strings
    
    Returns:
        Embedding vector(s) as numpy array or JAX array
    """
    model = _get_embedding_model()
    
    if model is None:
        # Return random embeddings as placeholder
        import numpy as np
        if isinstance(text, str):
            return np.random.randn(384).astype(np.float32)
        return np.random.randn(len(text), 384).astype(np.float32)
    
    try:
        # SentenceTransformer
        if hasattr(model, "encode"):
            return model.encode(text, convert_to_numpy=True)
        
        # HuggingFace transformers
        tokenizer = model["tokenizer"]
        encoder = model["model"]
        
        if isinstance(text, str):
            text = [text]
        
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        import torch
        with torch.no_grad():
            outputs = encoder(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()
    except Exception as e:
        print(f"[Trident Warning] Embedding failed: {e}")
        import numpy as np
        if isinstance(text, str):
            return np.random.randn(384).astype(np.float32)
        return np.random.randn(len(text), 384).astype(np.float32)


def similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.
    
    Returns:
        Similarity score between 0 and 1
    """
    import numpy as np
    
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    
    # Cosine similarity
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot / (norm1 * norm2))


# =============================================================================
# Tokenization
# =============================================================================

_default_tokenizer = None


def _get_tokenizer():
    """Get default tokenizer."""
    global _default_tokenizer
    
    if _default_tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _default_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            return None
    
    return _default_tokenizer


def tokenize(text: str, return_tokens: bool = False) -> Union[List[int], List[str]]:
    """
    Tokenize text.
    
    Args:
        text: Text to tokenize
        return_tokens: If True, return token strings; else return IDs
    
    Returns:
        List of token IDs or token strings
    """
    tokenizer = _get_tokenizer()
    
    if tokenizer is None:
        # Simple whitespace tokenization fallback
        tokens = text.split()
        if return_tokens:
            return tokens
        return list(range(len(tokens)))
    
    if return_tokens:
        return tokenizer.tokenize(text)
    return tokenizer.encode(text)


def detokenize(tokens: Union[List[int], List[str]]) -> str:
    """
    Convert tokens back to text.
    
    Args:
        tokens: Token IDs or token strings
    
    Returns:
        Decoded text
    """
    tokenizer = _get_tokenizer()
    
    if tokenizer is None:
        if isinstance(tokens[0], str):
            return " ".join(tokens)
        return "[Cannot detokenize without tokenizer]"
    
    if isinstance(tokens[0], str):
        return tokenizer.convert_tokens_to_string(tokens)
    return tokenizer.decode(tokens)


def count_tokens(text: str) -> int:
    """Count the number of tokens in text."""
    return len(tokenize(text))


# =============================================================================
# Information Extraction
# =============================================================================

def extract(
    text: str,
    schema: Dict[str, Any],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract structured information from text based on a schema.
    
    Args:
        text: Source text
        schema: Schema describing fields to extract
            Example: {"name": "string", "amount": "number", "items": "list"}
        model: Optional specific model to use
    
    Returns:
        Dictionary with extracted fields
    """
    # Build extraction prompt
    schema_str = json.dumps(schema, indent=2)
    prompt = f"""Extract the following fields from the text.
Return a JSON object matching this schema:
{schema_str}

Return ONLY valid JSON, no other text."""
    
    response = llm_query(text, prompt)
    
    # Try to parse JSON from response
    try:
        # Find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass
    
    # Return raw response as fallback
    return {"raw_response": response, "extraction_failed": True}


def classify(
    text: str,
    labels: List[str],
    multi_label: bool = False,
) -> Union[str, List[str]]:
    """
    Classify text into one or more labels.
    
    Args:
        text: Text to classify
        labels: List of possible labels
        multi_label: Allow multiple labels
    
    Returns:
        Single label or list of labels
    """
    labels_str = ", ".join(labels)
    
    if multi_label:
        prompt = f"Classify this text into one or more of these categories: {labels_str}. Return only the matching category names, comma-separated."
    else:
        prompt = f"Classify this text into exactly one of these categories: {labels_str}. Return only the category name."
    
    response = llm_query(text, prompt).strip()
    
    if multi_label:
        found = [label for label in labels if label.lower() in response.lower()]
        return found if found else [labels[0]]
    
    # Find best matching label
    response_lower = response.lower()
    for label in labels:
        if label.lower() in response_lower:
            return label
    
    return labels[0]  # Default to first label


def summarize(text: str, max_length: int = 100) -> str:
    """
    Summarize text.
    
    Args:
        text: Text to summarize
        max_length: Approximate maximum length of summary
    
    Returns:
        Summary text
    """
    prompt = f"Summarize the following text in approximately {max_length} words. Be concise and capture the main points."
    return llm_query(text, prompt)
