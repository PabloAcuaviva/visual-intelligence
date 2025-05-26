# Template Repository

This repository serves as a basic template to kickstart new projects. It includes essential configurations for maintaining code consistency.

## Features

- **Pre-commit Hooks**:
  - Configured to run `flake8`, `black`, and `isort` for linting, formatting, and organizing imports.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Install `pre-commit` in your environment:
   ```bash
   pip install pre-commit
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Tools and Configurations

- **Flake8**: Linting to catch potential issues and enforce coding standards.
- **Black**: Automatic code formatter for consistent style.
- **Isort**: Automatically sorts and organizes imports in Python files.

## How to Use

1. Use this template as the starting point for your project.

2. Create a new environment. We use here conda as an example.
```bash
conda create -n myenv python=3.12 --no-default-packages
conda install pip
```
3. Install pre-commit
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files # [Optional] Run against all files
```
4. Add additional configurations or tools as needed.
5. Modify `README.md` appropiately.

And your are finished!
