import multimodel_core

print("Testing multimodel_core...")
models, meta = multimodel_core.load_both_candidates("cpu")
print("Models loaded:", list(models.keys()))
for k, v in meta.items():
    print(f"{k}: {v['name']} (params: {v['parameters']})")

print("All multimodel_core tests passed.")
