# Visual-Intelligence

This repository contains the scripts required to reproduce the datasets used in the paper **"Rethinking Visual Intelligence: Insights From Video Pretraining"**.
It includes both **JSON** and **image** versions of the datasets, as well as the code to **decode images back into JSON** for downstream evaluation.

---

## Getting Started

1. Create a new environment (example using conda):

```bash
conda create -n visual-intelligence python=3.12 --no-default-packages
conda activate visual-intelligence
conda install pip
pip install -r requirements.txt
```

2. Generate datasets:

```bash
# ARC-related datasets
python3 generate_arc.py

# General visual intelligence tasks
python3 generate.py
```

3. _(Optional)_ To generate training videos for VDM models, run:

```bash
python3 generate_videos_for_tasks.py
```

## Contributing

We use pre-commit to maintain consistent code style.
Install and run it as follows:

```bash
pip install pre-commit
pre-commit install
# Optional: run checks against all files
pre-commit run --all-files
```
