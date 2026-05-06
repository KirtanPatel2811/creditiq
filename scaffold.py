# scaffold.py  — run once from the project root
import os

dirs = [
    "src/data",
    "src/features",
    "src/models",
    "src/explainability",
    "src/monitoring",
    "src/api",
    "src/app",
    "data/raw",
    "data/processed",
    "data/reference",
    "models",
    "notebooks",
    "tests",
    "reports",
    "mlruns",
    ".github/workflows",
]

files = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/loader.py",
    "src/data/validator.py",
    "src/data/splitter.py",
    "src/features/__init__.py",
    "src/features/engineer.py",
    "src/features/preprocessor.py",
    "src/models/__init__.py",
    "src/models/train.py",
    "src/models/evaluate.py",
    "src/models/tune.py",
    "src/explainability/__init__.py",
    "src/explainability/shap_explainer.py",
    "src/explainability/lime_explainer.py",
    "src/monitoring/__init__.py",
    "src/monitoring/drift_detector.py",
    "src/api/__init__.py",
    "src/api/main.py",
    "src/api/schemas.py",
    "src/app/__init__.py",
    "src/app/streamlit_app.py",
    "tests/__init__.py",
    "tests/test_features.py",
    "tests/test_model.py",
    "tests/test_api.py",
    "notebooks/01_eda.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "notebooks/03_modelling.ipynb",
    "notebooks/04_explainability.ipynb",
    ".github/workflows/ci.yml",
    "params.yaml",
    "dvc.yaml",
    ".gitignore",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

for f in files:
    if not os.path.exists(f):
        open(f, "w").close()

print("✅  Scaffold created.")
