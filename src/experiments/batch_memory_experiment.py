import torch
import pandas as pd
import gc
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.factory import ModelFactory
from src.decompositions.registry import DECOMPOSITION_REGISTRY
from src.decompositions.replacer import ModelReplacer

def measure_peak_memory(model, batch_size, input_shape=(3, 32, 32), device="cuda"):
    model.eval()
    model.to(device)
    dummy = torch.randn(batch_size, *input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
            
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        model(dummy)
        torch.cuda.synchronize()
        
    peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    del dummy
    return peak_mb

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA required for peak memory profiling.")
        return
        
    target_layers = [
        "features.0", "features.4", "features.8", "features.11", 
        "features.15", "features.18", "features.22", "features.25",
        "classifier.0", "classifier.3", "classifier.6"
    ]
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []
    
    print("Loading Baseline...")
    base_model = ModelFactory.get_model("vgg11_bn", num_classes=10, pretrained=True)
    base_model.to(device)
    
    for bs in batch_sizes:
        mem = measure_peak_memory(base_model, bs, device=device)
        print(f"Baseline (BS={bs}): {mem:.2f} MB")
        results.append({"method": "Baseline", "batch_size": bs, "peak_inference_memory_mb": mem})
    
    base_model.cpu()
    del base_model
    torch.cuda.empty_cache()
    
    configs = [
        ("SVD", 15, {}),
        ("Tucker", 15, {}),
        ("TT", 15, {}),
        ("CP", 15, {"cp_parafac_on_cpu": True})
    ]
    
    for algo, rank, kwargs in configs:
        print(f"Compressing with {algo} (rank={rank})...")
        model = ModelFactory.get_model("vgg11_bn", num_classes=10, pretrained=True)
        try:
            ModelReplacer.replace_layers(model, DECOMPOSITION_REGISTRY[algo], target_layers, rank=rank, **kwargs)
            model.to(device)
            for bs in batch_sizes:
                mem = measure_peak_memory(model, bs, device=device)
                print(f"{algo} (BS={bs}): {mem:.2f} MB")
                results.append({"method": algo, "batch_size": bs, "peak_inference_memory_mb": mem})
        except Exception as e:
            print(f"Error compressing with {algo}: {e}")
        
        model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../runs/run_20260523_151331/batch_memory_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
