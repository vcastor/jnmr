import os
import pickle

CACHE_DIR = "analysis/cache"

def save_cache(name, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(data, f)

def load_cache(name):
    with open(os.path.join(CACHE_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)
