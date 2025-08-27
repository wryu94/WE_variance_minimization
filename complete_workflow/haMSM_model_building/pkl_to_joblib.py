import pickle
from joblib import dump

# File name
name = 'dimreduce_model'

# Load existing pickle
with open(f"{name}.pkl", "rb") as f:
    dimreduce_model = pickle.load(f)

# Save as joblib
dump(dimreduce_model, f"{name}.joblib")

