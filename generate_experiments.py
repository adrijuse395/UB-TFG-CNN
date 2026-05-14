import json
import numpy as np

def get_ranks(min_val, max_val, num_samples):
    ranks = set()
    # Generate an excessive number of points to ensure enough unique integers
    pts = np.geomspace(min_val, max_val, num_samples * 10).astype(int)
    for p in pts:
        ranks.add(p)
    ranks = sorted(list(ranks))
    
    # We want exactly `num_samples` ranks. 
    if len(ranks) <= num_samples:
        return [int(x) for x in ranks]
    
    # Subsample exactly num_samples 
    indices = np.linspace(0, len(ranks) - 1, num_samples).astype(int)
    return [int(ranks[i]) for i in indices]

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Fix resource_limits and fine_tuning batches so we don't under-train or clamp ranks
    if "resource_limits" not in config:
        config["resource_limits"] = {}
    config["resource_limits"]["max_rank"] = 4000
    
    if "global_settings" in config and "fine_tuning" in config["global_settings"]:
        config["global_settings"]["fine_tuning"]["max_train_batches_per_epoch"] = 0
        config["global_settings"]["fine_tuning"]["max_val_batches_per_epoch"] = 0

    target_layers = [
        "features.0", "features.4", "features.8", "features.11",
        "features.15", "features.18", "features.22", "features.25",
        "classifier.0", "classifier.3", "classifier.6"
    ]

    methods_config = [
        ("SVD", 2, 400),
        ("Tucker", 2, 400),
        ("TT", 2, 400),
        ("CP", 2, 2000)
    ]

    num_samples = 10
    experiments = []

    for method, min_val, max_val in methods_config:
        ranks = get_ranks(min_val, max_val, num_samples)
        for r in ranks:
            # We add fine_tuning: true
            experiments.append({
                "name": f"{method} rank {r:04d} | ft",
                "method": method,
                "target_layers": target_layers,
                "rank": r,
                "fine_tuning": True
            })

    config['experiments'] = experiments

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Generated {len(experiments)} experiments.")
    print("Method breakdown:")
    for method, min_val, max_val in methods_config:
        ranks = get_ranks(min_val, max_val, num_samples)
        print(f"  - {method}: {len(ranks)} samples (min: {ranks[0]}, max: {ranks[-1]})")

if __name__ == '__main__':
    main()
