## Matrix Optimization
### Usage
After cloning the git repo in the main ReadMe that sets up the environment, add the files listed in the outlier_reducer folder to the fake_quant directory. Replace the requirements.txt files(this extension requires an upgrade in transformers) and install the requirements with the line of code: 
``` bash
pip install -r requirements.txt
```
The template for training the new rotation matrix model is as follows: 
```
python train_on_act.py --threshold {} --power {} --num_epochs {}
```
An example code of how to call it with values would be: 
```
python train_on_act.py --threshold {} --power {} --num_epochs {}
```
The generated matrix will be saved in the new saved_tensors directory. The matrix will me saved in a path that looks like the following: 
```
saved_tensors/orthogonal_matrix_epoch_0_T2.0_P2.0_B32.pt
```
Get the absolute path to the model. Copy this path and get the path to the model by calling realpath and entering the saved path from the above step. For this your line of code will look similar to the following: 

```
realpath saved_tensors/orthogonal_matrix_epoch_0_T2.0_P2.0_B32.pt

```
Copy and save the new path that should look similar to the following: 

```
/MatrixOptimize/fake_quant/saved_tensors/orthogonal_matrix_epoch_0_T2.0_P2.0_B32.pt
```
Navigate to rotation_utils.py file and replace the function def get_orthogonal_matrix with the following code below under function to change. Input the previously saved path to the matrix by replacing “path/to/matrix” with your path saved from the above step.   

Function to change: 
In the get_orthogonal_matrix() function in the rotation_utils.py of the original repo, change the code like this:
``` python
def get_orthogonal_matrix(size, mode, device=utils.DEV):
    path_to_matrix = "path/to/matrix"
    matrix = torch.load(path_to_matrix).to(device=device).to(dtype=torch.float64)
    return matrix
```
