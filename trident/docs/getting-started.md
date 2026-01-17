# Getting Started with Trident

Welcome to Trident! This guide will help you build your first AI-native document processing pipeline in under 5 minutes.

## 1. Installation

Trident requires **Python 3.9+**.

```bash
pip install trident-lang
```

**Note for TPU Users:**
If you are running on a TPU VM (Google Cloud), Trident will automatically detect the TPU. No extra configuration is needed.

## 2. Your First Pipeline

Trident is designed to be readable. Let's solve a real problem: **extracting data from an invoice**.

Create a file named `hello_trident.tri`:

```trident
pipeline HelloTrident:
    fn main():
        print("Hello, AI World!")

        # Built-in tensor creation
        let t = [1, 2, 3]
        print(t * 2)
```

Run it:

```bash
trident run hello_trident.tri
```

## 3. The Power of Primitives

Trident's magic lies in its primitives: `vision`, `ocr`, and `nlp`. You don't import huge libraries; you just use the keywords.

Here is a complete invoice processor:

```trident
pipeline InvoiceProcessor:
    fn process_invoice(path: string):
        # Read: No PIL or OpenCV imports needed
        let img = vision.read_file(path)

        # OCR: Native keyword
        let text = ocr.extract(img)

        # NLP: Native keyword for extraction
        let info = nlp.extract(text, schema={
            "vendor": string,
            "amount": float
        })

        print(info)
```

## 4. Next Steps

- Check out the [Tutorials](./tutorials/invoice-processing.md) for a deep dive.
- Learn about [Trident's Type System](./concepts/types.md).
- deploy your pipeline with `trident compile`.
