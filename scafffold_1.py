# run from project root: python -c "from src.data.loader import load_dataset; ..."
from src.data.loader import load_dataset
from src.data.splitter import split_data
from src.features.preprocessor import build_preprocessor
import numpy as np

X, y = load_dataset("german_credit")
print(f"Shape: {X.shape}, Default rate: {y.mean():.1%}")

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

X_train_t = preprocessor.fit_transform(X_train)
X_val_t   = preprocessor.transform(X_val)

print(f"Transformed train shape: {X_train_t.shape}")
print(f"No NaNs in train: {not np.isnan(X_train_t).any()}")
print("✅  Phase 1 pipeline validated")