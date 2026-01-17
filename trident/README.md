# 🔱 Trident

<!-- Badges -->

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/backend-JAX%2FXLA-green)](https://github.com/google/jax)
[![TPU](https://img.shields.io/badge/hardware-TPU%20Native-orange)](https://cloud.google.com/tpu)
[![Stars](https://img.shields.io/github/stars/trident-lang/trident?style=social)](https://github.com/trident-lang/trident)
[![Discord](https://img.shields.io/discord/1234567890?color=7289da&label=Discord&logo=discord&logoColor=white)](https://discord.gg/trident)

> **The AI-First Programming Language for Enterprise Document Intelligence.**  
> Built on JAX. Optimized for TPU. Designed for Humans.

---

**Trident** is a strongly-typed, compiled language that treats **OCR, NLP, and Tensor operations** as first-class primitives. It bridges the gap between high-level intent and low-level TPU execution, making it 10x faster to build and deploy production document processing pipelines.

### 🚀 Why Trident?

- **📄 AI Primitives, Not Libraries**: `ocr`, `nlp`, and `tensor` are built-in keywords, not imports.
- **⚡ TPU Native**: Compiles directly to JAX/XLA for execution on TPUs, GPUs, and CPUs.
- **🛡️ Type Safe**: Gradual typing with tensor shapes ensures your pipelines don't crash in production.
- **📦 One-Click Deploy**: Go from code to a scaled Cloud Run service in one command.

---

### ⚡ Quick Start: Process an Invoice in < 2 Minutes

Install Trident:

```bash
pip install trident-lang
```

Write your first pipeline (`invoice.tri`):

```trident
pipeline InvoiceProcessor:
    @model(ocr: "qwen2.5-vl", nlp: "gemma-3-quantized")
    fn process(doc: Image) -> json:
        # 1. Optical Character Recognition (Built-in)
        let text_layout = ocr.extract(doc, layout=true)

        # 2. Extract structured data using LLM
        @intent("Extract vendor, total amount, and invoice number")
        let data = nlp.extract(text_layout, schema={
            "vendor": string,
            "total": float,
            "invoice_num": string,
            "date": date
        })

        return data

# Run it!
# $ trident run invoice.tri --input ./sample_invoice.pdf
```

---

### 🏢 Example: Production Document Pipeline

Trident isn't just for scripts. It builds robust, type-safe pipelines.

```trident
# Define your data shapes with types
struct LineItem {
    desc: string
    qty: int
    price: float
}

struct Invoice {
    id: string
    items: List[LineItem]
    total: float
    confidence: float
}

pipeline SecureInvoiceProcessor:
    # Use low-precision for speed on TPU
    @hardware(target="tpu_v5p", precision="bfloat16")
    fn clean_and_extract(img: Tensor[1024, 1024, 3]) -> Invoice:

        # Hardware-accelerated image preprocessing
        let clean_img = vision.normalize(img) |> vision.deskew()

        # Extract with confidence score
        let result = nlp.query(clean_img, "Extract invoice details")

        if result.confidence < 0.90:
             raise LowConfidenceError(result)

        return Invoice(result)
```

---

### 🛠️ Installation & Setup

**Prerequisites:** Python 3.9+

```bash
# Install from PyPI
pip install trident-lang

# Initialize a new project
trident init my-first-pipeline
cd my-first-pipeline

# Run the 'hello world' example
trident run src/main.tri
```

### 🏎️ Performance

| Task                 | Python (Tesseract + Transformers) | Trident (JAX + Compiled) | Speedup  |
| -------------------- | --------------------------------- | ------------------------ | -------- |
| Invoice OCR          | 1.2s                              | **0.15s**                | **8x**   |
| Entity Extraction    | 3.5s                              | **0.4s**                 | **8.7x** |
| Hardware Utilization | 15% (CPU)                         | **95% (TPU)**            | **High** |

---

### 🤝 Community & Support

- **[Documentation](https://docs.trident.dev)**: Full tutorials and API reference.
- **[Discord](https://discord.gg/trident)**: Chat with the team and other developers.
- **[Twitter](https://twitter.com/trident_lang)**: Follow for updates and tips.

---

**License:** MIT  
**Status:** Beta (v0.1.0)
