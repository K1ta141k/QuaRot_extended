## Matrix Optimization
### Usage
After cloning the git repo in the main ReadMe that sets up the environment, install all the requirements that are listed in the main directory
``` bash
pip install -r requirements.txt
python train_on_act.py --threshold {} --power {} --num_epochs {}
```
The generated matrix will be saved in the new saved_tensors directory. Get the absolute path to the model. In the get_orthogonal_matrix() function in the rotation_utils.py of the original repo, change the code like this:
``` python
def get_orthogonal_matrix(size, mode, device=utils.DEV):
    path_to_matrix = "path/to/matrix"
    matrix = torch.load(path_to_matrix).to(device=device).to(dtype=torch.float64)
    return matrix
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')
```
