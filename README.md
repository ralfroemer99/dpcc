This repository contains the code to our paper [Diffusion Predictive Control with Constraints](https://arxiv.org/abs/2412.09342). We build upon the temporal U-Net implementation from [Diffuser](https://github.com/jannerm/diffuser) and use the Avoiding environment from [D3IL](https://github.com/ALRhub/d3il).

![alt text](https://github.com/ralfroemer99/dpcc/blob/main/figures/avoiding.png?raw=true)

## Installation
Clone the repo and run:
```bash
conda create -n dpcc python=3.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
You also need to install [D3IL](https://github.com/ALRhub/d3il) for the simulation environment.

## Training
To train the diffusion policy, run:
```bash
python scripts/train.py
```
You can also visualize the training data without constraints and the novel test-time constraints:
```bash
python scripts/visualize_data_constraints/train.py
```

## Testing
To evaluate DPCC and reproduce the results reported in the paper, run:
```bash
python scripts/eval.py
python scripts/load_results.py
```
