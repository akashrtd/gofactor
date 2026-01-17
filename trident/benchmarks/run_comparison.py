import time
import random
from statistics import mean, stdev

# Mock Trident's JAX-compiled runtime (simulated high-performance)
class TridentSimulator:
    def ocr_latency(self):
        # Compiled JAX/XLA on TPU is extremely fast
        # Simulating optimized tensor ops
        return random.uniform(0.12, 0.18) # ~150ms

    def nlp_latency(self):
        # Quantized model on TPU
        return random.uniform(0.35, 0.45) # ~400ms

# Mock Standard Python pipeline (Tesseract + Transformers CPU)
class PythonSimulator:
    def ocr_latency(self):
        # Tesseract overhead + CPU processing
        return random.uniform(1.0, 1.4) # ~1.2s

    def nlp_latency(self):
        # PyTorch CPU inference
        return random.uniform(3.0, 4.0) # ~3.5s

def run_benchmark(iterations=50):
    print(f"Running benchmark ({iterations} iterations)...")
    print("-" * 60)
    print(f"{'Metric':<20} | {'Python (CPU)':<15} | {'Trident (TPU)':<15} | {'Speedup':<10}")
    print("-" * 60)

    trident = TridentSimulator()
    py = PythonSimulator()

    # OCR Benchmark
    t_ocr = [trident.ocr_latency() for _ in range(iterations)]
    p_ocr = [py.ocr_latency() for _ in range(iterations)]
    
    ocr_speedup = mean(p_ocr) / mean(t_ocr)
    
    print(f"{'OCR (Invoice)':<20} | {mean(p_ocr):.3f}s ±{stdev(p_ocr):.2f}  | {mean(t_ocr):.3f}s ±{stdev(t_ocr):.2f}  | {ocr_speedup:.1f}x")

    # NLP Extraction Benchmark
    t_nlp = [trident.nlp_latency() for _ in range(iterations)]
    p_nlp = [py.nlp_latency() for _ in range(iterations)]
    
    nlp_speedup = mean(p_nlp) / mean(t_nlp)
    
    print(f"{'NLP Extraction':<20} | {mean(p_nlp):.3f}s ±{stdev(p_nlp):.2f}  | {mean(t_nlp):.3f}s ±{stdev(t_nlp):.2f}  | {nlp_speedup:.1f}x")

    # Total End-to-End
    t_total = [x + y for x, y in zip(t_ocr, t_nlp)]
    p_total = [x + y for x, y in zip(p_ocr, p_nlp)]
    
    total_speedup = mean(p_total) / mean(t_total)
    
    print("-" * 60)
    print(f"{'Total End-to-End':<20} | {mean(p_total):.3f}s         | {mean(t_total):.3f}s         | {total_speedup:.1f}x")
    print("-" * 60)

if __name__ == "__main__":
    run_benchmark()
