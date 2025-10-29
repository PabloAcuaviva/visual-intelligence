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

3. To generate training videos for VDM models, run:

```bash
python3 generate_videos_for_tasks.py
```

## Finetuning Models

### Finetuning LLMs
See the companion repository for instructions: [llm visual intelligence](https://github.com/PabloAcuaviva/llm-visual-intelligence).

### Finetuning VDMs

Finetuning Video Diffusion Models (VDMs) requires preparing datasets in video format and leveraging the training utilities provided by the respective base model repositories.

**Supported base models and references**

* **CogVideoX 1.5:** Experimental code is provided in [CogVideoX1.5 Experimental](https://github.com/PabloAcuaviva/visual-intelligence-cog-video)`.  
  Upstream repository: [CogVideoX](https://github.com/zai-org/CogVideo)  
> ⚠️ **Warning:**  
> The experimental code provided here is primarily for reference. It was used during experiments and includes additional functionality and unfinished features that did not make it into the paper. It has not been cleaned or documented and is **NOT RECOMMENDED for general use**.  
> For practical finetuning, please use the official [CogVideoX Repository](https://github.com/zai-org/CogVideo).

* **Wan2.1:** For finetuning, we recommend using either [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) (used in our experiments) or [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), both of which provide compatible and well-maintained finetuning pipelines.

* **LTX:** Follow the official implementation for finetuning and dataset preparation.  
  Repository: [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer)


## Contributing

We use pre-commit to maintain a consistent code style.
Install and run it as follows:

```bash
pip install pre-commit
pre-commit install
# Optional: run checks against all files
pre-commit run --all-files
```
