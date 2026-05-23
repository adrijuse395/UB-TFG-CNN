import csv

with open('/home/adria/Desktop/TFG/projects/CNN/runs/run_20260520_184504/results.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

baseline = next(r for r in rows if r['method'] in ('', 'None'))
print(f"Baseline Accuracy: {baseline['accuracy']}%, Params: {baseline['total_parameters']}")

for method in ["SVD", "Tucker", "TT", "CP"]:
    method_rows = [r for r in rows if r['method'] == method]
    if not method_rows:
        continue
    
    method_rows.sort(key=lambda r: int(r['total_parameters']))
    
    print(f"\n--- {method} ---")
    n = len(method_rows)
    indices = [0, n//4, n//2, 3*n//4, n-1]
    
    for idx in indices:
        r = method_rows[idx]
        print(f"Rank: {r['rank']}, Params: {r['total_parameters']}, CR: {float(r['compression_ratio']):.2f}x, Acc: {float(r['accuracy']):.2f}%")
