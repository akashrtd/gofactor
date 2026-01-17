"""
Vision and OCR primitives for Trident.

Provides built-in operations for:
- Image loading and preprocessing
- OCR text extraction
- Document layout analysis
- Vision-language model integration
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import io


@dataclass
class Image:
    """
    Image type for Trident.
    
    Wraps image data with metadata for processing.
    """
    data: Any  # PIL Image or numpy array
    path: Optional[str] = None
    width: int = 0
    height: int = 0
    channels: int = 3
    format: str = "RGB"
    
    def __post_init__(self) -> None:
        """Extract dimensions from data if available."""
        if hasattr(self.data, "size"):
            self.width, self.height = self.data.size
        elif hasattr(self.data, "shape"):
            if len(self.data.shape) >= 2:
                self.height, self.width = self.data.shape[:2]
                if len(self.data.shape) == 3:
                    self.channels = self.data.shape[2]
    
    def to_numpy(self) -> Any:
        """Convert to numpy array."""
        import numpy as np
        
        if hasattr(self.data, "__array__"):
            return np.asarray(self.data)
        return self.data
    
    def to_pil(self) -> Any:
        """Convert to PIL Image."""
        try:
            from PIL import Image as PILImage
            
            if isinstance(self.data, PILImage.Image):
                return self.data
            
            import numpy as np
            arr = np.asarray(self.data)
            return PILImage.fromarray(arr)
        except ImportError:
            raise RuntimeError("PIL not available")


@dataclass
class Document:
    """
    Document type for OCR output.
    
    Contains extracted text with optional layout information.
    """
    text: str
    pages: List[Dict[str, Any]] = field(default_factory=list)
    layout: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    language: str = "auto"
    
    def __str__(self) -> str:
        return self.text
    
    def get_blocks(self) -> List[Dict[str, Any]]:
        """Get text blocks with positions."""
        if self.layout and "blocks" in self.layout:
            return self.layout["blocks"]
        return []
    
    def get_tables(self) -> List[Dict[str, Any]]:
        """Get detected tables."""
        if self.layout and "tables" in self.layout:
            return self.layout["tables"]
        return []


# =============================================================================
# Image Loading
# =============================================================================

def read_image(path: str) -> Image:
    """
    Load an image from file.
    
    Args:
        path: Path to image file
    
    Returns:
        Image object
    """
    try:
        from PIL import Image as PILImage
        
        img = PILImage.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        return Image(data=img, path=path)
    except ImportError:
        # Fallback without PIL
        return Image(data=None, path=path)


def read_pdf(path: str) -> List[Image]:
    """
    Load pages from a PDF as images.
    
    Args:
        path: Path to PDF file
    
    Returns:
        List of Image objects (one per page)
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
            
            from PIL import Image as PILImage
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(Image(data=img, path=f"{path}#page{page_num + 1}"))
        
        return images
    except ImportError:
        raise RuntimeError("PyMuPDF not available for PDF processing")


def preprocess_image(
    image: Image,
    size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
) -> Image:
    """
    Preprocess an image for model input.
    
    Args:
        image: Input image
        size: Target size (width, height) or None to keep original
        normalize: Whether to normalize pixel values
    
    Returns:
        Preprocessed image
    """
    try:
        from PIL import Image as PILImage
        import numpy as np
        
        pil_img = image.to_pil()
        
        if size:
            pil_img = pil_img.resize(size, PILImage.LANCZOS)
        
        if normalize:
            arr = np.array(pil_img).astype(np.float32) / 255.0
            return Image(data=arr, path=image.path)
        
        return Image(data=pil_img, path=image.path)
    except Exception as e:
        print(f"[Trident Warning] Image preprocessing failed: {e}")
        return image


# =============================================================================
# OCR Operations
# =============================================================================

_ocr_model = None
_ocr_processor = None


def _get_ocr_model():
    """Get or initialize the OCR model."""
    global _ocr_model, _ocr_processor
    
    if _ocr_model is None:
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            model_id = "Qwen/Qwen2-VL-2B-Instruct"
            _ocr_processor = AutoProcessor.from_pretrained(model_id)
            _ocr_model = AutoModelForVision2Seq.from_pretrained(model_id)
        except Exception as e:
            print(f"[Trident Warning] Could not load OCR model: {e}")
            return None, None
    
    return _ocr_model, _ocr_processor


def ocr_extract(
    image: Union[Image, str],
    layout: bool = False,
    language: str = "auto",
) -> Document:
    """
    Extract text from an image using OCR.
    
    Args:
        image: Image object or path to image
        layout: Whether to include layout information
        language: Language hint (auto-detect if "auto")
    
    Returns:
        Document with extracted text
    """
    # Handle string path
    if isinstance(image, str):
        image = read_image(image)
    
    # Try transformer-based OCR
    model, processor = _get_ocr_model()
    
    if model is not None and processor is not None:
        return _ocr_with_vlm(image, model, processor, layout)
    
    # Fallback to Tesseract
    return _ocr_with_tesseract(image, layout, language)


def _ocr_with_vlm(image: Image, model, processor, layout: bool) -> Document:
    """OCR using vision-language model."""
    try:
        pil_image = image.to_pil()
        
        # Prepare prompt for OCR
        prompt = "Extract all text from this image. Preserve formatting and layout."
        if layout:
            prompt += " Also identify text blocks, tables, and their positions."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil_image], return_tensors="pt")
        
        outputs = model.generate(**inputs, max_new_tokens=2048)
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract the response (after the prompt)
        extracted_text = decoded.split(prompt)[-1].strip()
        
        return Document(
            text=extracted_text,
            source=image.path,
            layout={"raw_response": decoded} if layout else None,
        )
    except Exception as e:
        print(f"[Trident Warning] VLM OCR failed: {e}")
        return _ocr_with_tesseract(image, layout, "auto")


def _ocr_with_tesseract(image: Image, layout: bool, language: str) -> Document:
    """Fallback OCR using Tesseract."""
    try:
        import pytesseract
        
        pil_image = image.to_pil()
        
        if layout:
            # Get full data including positions
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            text = pytesseract.image_to_string(pil_image)
            
            blocks = []
            for i, txt in enumerate(data["text"]):
                if txt.strip():
                    blocks.append({
                        "text": txt,
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                        "confidence": data["conf"][i],
                    })
            
            return Document(
                text=text,
                source=image.path,
                layout={"blocks": blocks},
            )
        else:
            text = pytesseract.image_to_string(pil_image)
            return Document(text=text, source=image.path)
    except ImportError:
        return Document(
            text="[OCR not available - install pytesseract]",
            source=image.path,
        )


def ocr_layout(image: Union[Image, str]) -> Dict[str, Any]:
    """
    Analyze document layout without full OCR.
    
    Returns detected regions: text blocks, tables, figures, etc.
    """
    doc = ocr_extract(image, layout=True)
    return doc.layout or {}


# =============================================================================
# Image Analysis
# =============================================================================

def describe_image(image: Union[Image, str], prompt: str = "Describe this image.") -> str:
    """
    Generate a description of an image using a vision-language model.
    
    Args:
        image: Image to describe
        prompt: Custom prompt for description
    
    Returns:
        Text description
    """
    if isinstance(image, str):
        image = read_image(image)
    
    model, processor = _get_ocr_model()  # Reuse VLM
    
    if model is None:
        return "[Vision model not available]"
    
    try:
        pil_image = image.to_pil()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil_image], return_tensors="pt")
        
        outputs = model.generate(**inputs, max_new_tokens=512)
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return decoded.split(prompt)[-1].strip()
    except Exception as e:
        return f"[Error describing image: {e}]"
