This repository contains the code to our paper "Diffusion Predictive Control with Constraints".

## Installation
```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

# Training
To train the trajectory diffusion model, run:
```bash
python scripts/train.py
```

# Testing
To evaluate DPCC and reproduce the results reported in the paper, run:
```bash
python scripts/eval.py
python scripts/load_results.py
```
