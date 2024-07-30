## Install on Ubuntu

```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

To create the custom Pointmaze datasets with 50Hz control frequency:
```bash
python diffuser/datasets/minari-dataset-generation/scripts/pointmaze/create_pointmaze_dataset.py
```
