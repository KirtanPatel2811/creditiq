# Contributing to CreditIQ

## Setup
```bash
conda create -n creditiq python=3.10 -y
conda activate creditiq
pip install -r requirements.txt
```

## Before committing
```bash
pytest tests/ -v          # all 13 tests must pass
python -m src.models.train  # verify pipeline runs
```

## Branch naming
- `feat/` — new features
- `fix/` — bug fixes
- `docs/` — documentation only
