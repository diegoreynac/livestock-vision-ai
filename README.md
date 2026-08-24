# livestock-vision-ai
Deep Learning and Computer Vision for Precision Livestock Farming

## Environment and dependency pinning

This repository uses the single canonical dependency file at `requirements.txt`.

Pinned versions for the current architecture stack:
- torch==2.1.0
- torchvision==0.16.0
- ultralytics==8.4.126
- numpy==1.25.2

These versions are intentionally pinned because the selected PyTorch 2.1.0 / torchvision 0.16.0 binaries are not compatible with NumPy 2.x in this environment. We observed that ultralytics installation and PyTorch extension loading can trigger NumPy 2.x warnings and runtime incompatibilities, so this repo keeps NumPy at 1.25.2 for reproducibility.
