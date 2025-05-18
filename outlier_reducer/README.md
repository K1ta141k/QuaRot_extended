## Matrix Optimization
### Usage

If you have not completed the instructions in the main readme file to set up the environment then set up an environment with python=3.11 (this can be done using conda) and clone the main repository with the original code and install the original requirements with the instructions below
```
git clone <https://github.com/K1ta141k/QuaRot.git>
cd QuaRot
pip install -r requirements.txt
pip install -e . 
```
While inside the QuaRot directory,  clone the QuaRot_extended repo with the following code

```
git clone <https://github.com/K1ta141k/QuaRot_extended.git>
```
Move to the QuaRot_extended repo and navigate to the outlier reducer with the following

```
cd QuaRot_extended
cd outlier_reducer 
```
Install the new requirements to upgrade in the transformer version with the line below
```
pip install -r requirements.txt 
```
You can now train the matrix. The template for training the new rotation matrix model is as follows:
```
python train_on_act.py --threshold {} --power {} --num_epochs {}
```
An example code of how to call it with values would be:
```
python train_on_act.py --threshold 2 --power 2 --num_epochs 1
```
The generated matrix will be saved in the new saved_tensors directory. The matrix will me saved in a path that looks like the following:
```
Matrix saved to saved_tensors/orthogonal_matrix_epoch_0_T2.0_P2.0_B32.pt
```
Get the absolute path to the model. Copy this path and get the path to the model by calling realpath and entering the saved path from the above step. For this your line of code will look similar to the following:

```
realpath saved_tensors/orthogonal_matrix_epoch_0_T2.0_P2.0_B32.pt
```
Copy and save the new path that is shown. This should look similar to the following:
```
/root/QuaRot/QuaRot_extended/outlier_reducer/saved_tensors/orthogonal_matrix_epoch_0_T2.0_P2.0_B32.pt
```
Once you have the path saved, navigate back to the original repository and reinstall the older version of transformers by running the code below once in the QuaRot directory
```
cd ../..
pip install -r requirements.txt 
```
Navigate to the fake_quant directory with the following
```
cd fake_quant 
```
You can edit the rotation_u-tils.py with a text editor such as nano or vim. With nano you can navigate to rotation_utils with the code
```
nano rotation_utils.py 
```
Edit the rotation_utils.py file and replace the function get_orthogonal_matrix() with the following code below. You can switch the code to the following:
```
def get_orthogonal_matrix(size, mode, device=utils.DEV):
    path_to_matrix = "path/to/matrix"
    matrix = torch.load(path_to_matrix).to(device=device).to(dtype=torch.float64)
    return matrix
```
Paste the previously saved path to the matrix by replacing “path/to/matrix” with your path saved from the above step. For example
```
def get_orthogonal_matrix(size, mode, device=utils.DEV):
    path_to_matrix = "/root/QuaRot/QuaRot_extended/outlier_reducer/saved_tensors/orthogonal_matrix_epoch_0_T2.0_P2.0_B32.pt"
    matrix = torch.load(path_to_matrix).to(device=device).to(dtype=torch.float64)
    return matrix
```
Now once in fake_quant you run the script from the original QuaRot code to check the perplexity. For example, to run the perplexity of LLaMA2-7B model with quantizing all weights and activations, you can run the following command:
```
python main.py --model meta-llama/Llama-2-7b-hf  --rotate --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --w_clip
```

The arguments are
* -model: the model name (or path to the weights)
* -bsz: the batch size for PPL evaluation
* -rotate: whether we want to rotate the model
* -lm_eval: whether we want to run LM-Eval for Zero-Shot tasks
* -tasks: the tasks for LM-Eval
* -cal_dataset: the calibration dataset for GPTQ quantization
* -a_bits: the number of bits for activation quantization
* -w_bits: the number of bits for weight quantization
* -v_bits: the number of bits for value quantization
* -k_bits: the number of bits for key quantization
* -w_clip: Whether we want to clip the weights
* -a_clip_ratio: The ratio of clipping for activation
* -k_clip_ratio: The ratio of clipping for key
* -v_clip_ratio: The ratio of clipping for value
* -w_asym: Whether we want to use asymmetric quantization for weights
* -a_asym: Whether we want to use asymmetric quantization for activation
* -v_asym: Whether we want to use asymmetric quantization for value
* -k_asym: Whether we want to use asymmetric quantization for key
* -a_groupsize: The group size for activation quantization
* -w_groupsize: The group size for weight quantization
* -v_groupsize: The group size for value quantization
* -k_groupsize: The group size for key quantization
