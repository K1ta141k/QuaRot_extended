## QuaRot Extensions
Ali Bauyrzhan, Dacia John, Abhinaya Menon

Note: There are two extensions that were implemented and they have been implemented on different requirements therefore it is suggested that you create two separate environments when testing each of the extensions. 

We have reproduced the paper results and have extended added two implementations of our custom methodologies. The detailed information is in the slides and in the pdf reports.

## Requirements for running Llama-2-7b-hf
GPU with at least 25GB GPU Memory, 11 TFLOPS.
Works properly on this image: https://hub.docker.com/r/cheyam/template3
## Environment setup
Create and activate a conda environment with python=3.11.
The original repository has some issues with running and we had to fix them. It's recommended to use the forked version with our fixes.
```bash
git clone https://github.com/K1ta141k/QuaRot.git
cd QuaRot
pip install -r requirements.txt
pip install -e .  # or pip install .
```
## Matrix optimization for outlier reduction
Currently, our methodology uses only the first activation layer for the data. To test the matrix, go to outlier_reducer directory.

## Skew based quantization mode selection
To test the methodology, go the the skew_based directory.
