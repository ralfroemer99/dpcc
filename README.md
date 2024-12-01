This repository contains the code to our paper "Diffusion Predictive Control with Constraints". We build upon the temporal U-Net implementation from [Diffuser](https://github.com/jannerm/diffuser) and use the Avoiding environment from [D3IL](https://github.com/ALRhub/d3il).

![alt text](https://github.com/ralfroemer99/dpcc/blob/main/figures/avoiding.png?raw=true)

## Installation
Clone our repo and run:
```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
You also need to install [D3IL](https://github.com/ALRhub/d3il) for the simulation environment.

## Training
To train the trajectory diffusion model, run:
```bash
python scripts/train.py
```

## Testing
To evaluate DPCC and reproduce the results reported in the paper, run:
```bash
python scripts/eval.py
python scripts/load_results.py
```
