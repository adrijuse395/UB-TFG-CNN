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
    
    svd_ranks = [2, 5, 7, 10, 12, 15, 18, 20, 23, 26, 30, 33, 39, 45, 50, 58, 64, 75, 88, 97, 113, 126, 147, 163, 190, 223, 251, 300, 337, 400]
    tucker_ranks = [2, 5, 8, 11, 13, 16, 19, 23, 27, 31, 37, 40, 47, 54, 61, 70, 81, 92, 104, 114, 128, 147, 167, 188, 213, 239, 259, 302, 349, 400]
    tt_ranks = [2, 4, 7, 9, 11, 14, 16, 18, 21, 25, 28, 31, 38, 42, 48, 57, 64, 72, 86, 96, 108, 129, 146, 164, 185, 221, 249, 285, 350, 400]
    cp_ranks = [2, 5, 8, 11, 14, 17, 20, 25, 28, 34, 40, 49, 58, 70, 84, 100, 120, 143, 171, 204, 243, 290, 326, 389, 465, 576, 732, 931, 1176, 1500]
    
    configs = [("SVD", r, {}) for r in svd_ranks] + \
              [("Tucker", r, {}) for r in tucker_ranks] + \
              [("TT", r, {}) for r in tt_ranks] + \
              [("CP", r, {"cp_parafac_on_cpu": True}) for r in cp_ranks]
    
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
