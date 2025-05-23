## Skew Based Quantization Mode Selection
### Usage
**Instructions for skew logic: **

If you have not completed the instructions in the main readme file to set up the environment then set up an environment with python=3.11 (this can be done using conda) and clone the main repository with the original code and install the original requirements with the instructions below
```
git clone https://github.com/K1ta141k/QuaRot.git
cd QuaRot
pip install -r requirements.txt
pip install -e . 
```
While inside the QuaRot directory,  clone the QuaRot_extended repo with the following code

```
git clone https://github.com/K1ta141k/QuaRot_extended.git
```
Move to the QuaRot_extended repo and navigate to skew_based with the following

```
cd QuaRot_extended
cd skew_based 
```

Navigate to the fake_quant directory and replace the files in the fake_quant directory with these files that has the corresponding names from the skew_based directory. These files are main.py, data_utils.py, skew_utils.py, utils.py. You can replace these files with the following template: 

```
cp /path/to/source/main.py /path/to/destination/main.py
```

An example of the command used to replace files would look like the following 
```
cp /root/QuaRot/QuaRot_extended/skew_based/main.py /root/QuaRot/fake_quant/main.py 
cp /root/QuaRot/QuaRot_extended/skew_based/data_utils.py /root/QuaRot/fake_quant/data_utils.py 
cp /root/QuaRot/QuaRot_extended/skew_based/skew_utils.py /root/QuaRot/fake_quant/skew_utils.py 
cp /root/QuaRot/QuaRot_extended/skew_based/utils.py /root/QuaRot/fake_quant/utils.py 
```

The skew logic can be tested by adding the following flag in addition to calling the original code: 
‘—a_auto_asym’ this implements the skew logic that automatically deciphers between the type of quantization 


To run the perplexity of `LLaMA2-7B` model with quantizing all weights and activations while using the skew logic, you can run the following command:
```
python main.py --model meta-llama/Llama-2-7b-hf  --rotate --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --w_clip --a_auto_asym 
```

Original QuaRot Readme instructions: 

# Fake Quantization in QuaRot
In this directory, we provide the torch scripts for the experiments in QuaRot. 

## Language Generation and Zero-Shot Evaluations
Currently, we only support **LLaMa-2** models. You can simply run the `main.py` to reproduce the results in the paper. The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--bsz`: the batch size for PPL evaluation
- `--rotate`: whether we want to rotate the model
- `--lm_eval`: whether we want to run LM-Eval for Zero-Shot tasks
- `--tasks`: the tasks for LM-Eval
- `--cal_dataset`: the calibration dataset for GPTQ quantization
- `--a_bits`: the number of bits for activation quantization
- `--w_bits`: the number of bits for weight quantization
- `--v_bits`: the number of bits for value quantization
- `--k_bits`: the number of bits for key quantization
- `--w_clip`: Whether we want to clip the weights
- `--a_clip_ratio`: The ratio of clipping for activation
- `--k_clip_ratio`: The ratio of clipping for key
- `--v_clip_ratio`: The ratio of clipping for value
- `--w_asym`: Whether we want to use asymmetric quantization for weights
- `--a_asym`: Whether we want to use asymmetric quantization for activation
- `--v_asym`: Whether we want to use asymmetric quantization for value
- `--k_asym`: Whether we want to use asymmetric quantization for key
- `--a_groupsize`: The group size for activation quantization
- `--w_groupsize`: The group size for weight quantization
- `--v_groupsize`: The group size for value quantization
- `--k_groupsize`: The group size for key quantization
  
For example, to run the perplexity of `LLaMA2-7B` model with quantizing all weights and activations, you can run the following command:

```bash
/bin/python main.py --model meta-llama/Llama-2-7b-hf  --rotate --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --w_clip
```



