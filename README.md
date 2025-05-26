# Visual-logic

## Contribute

1. Create a new environment. We use here conda as an example.
```bash
conda create -n visual-logic python=3.12 --no-default-packages
conda install pip
```

2. Install pre-commit
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files # [Optional] Run against all files
```

TODO: Add conda .yaml instead of using a completely new environment (to install dependencies)
