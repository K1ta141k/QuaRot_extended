import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from typing import List
import torch.nn as nn
import sys
from torch.cuda.amp import autocast, GradScaler
import json
from datetime import datetime
import torch.nn.functional as F
import argparse
import random
import transformers
import datasets
import copy
from torch.utils.data import Dataset
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hadamard_utils
from cayley_optimizer import SGDG
from torch.utils.data import TensorDataset, DataLoader
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


# Enable PyTorch optimizations
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cuDNN

class BaseLLaMAFirstLayer(nn.Module):
    """
    The first layer of Llama-2-7b-hf. 
    """
    def __init__(self, base_model, device='cuda'):
        super().__init__()
        self.config = base_model.config
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Ensure components are in float32
        self.embed_tokens = copy.deepcopy(base_model.model.embed_tokens).to(device).float()
        self.first_layer = copy.deepcopy(base_model.model.layers[0]).to(device).float()
        self.rotary_emb = LlamaRotaryEmbedding(self.config).to(device).float()

    def forward(self, input_ids):
        # Handle input shape: [batch_size, num_chunks, seq_len] -> [batch_size * num_chunks, seq_len]
        if len(input_ids.shape) == 3:
            batch_size, num_chunks, seq_len = input_ids.shape
            input_ids = input_ids.reshape(-1, seq_len)
        else:
            batch_size, seq_len = input_ids.shape

        device = input_ids.device

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Rotary embeddings
        dummy_tensor = torch.zeros(
            (batch_size, self.num_heads, seq_len, self.head_dim),
            device=device,
            dtype=hidden_states.dtype
        )
        cos, sin = self.rotary_emb(dummy_tensor, position_ids=position_ids)

        # Forward through the first transformer layer
        hidden_states = self.first_layer(hidden_states, position_embeddings=(cos, sin))[0]

        return hidden_states

class RotatedLLaMAFirstLayer(nn.Module):
    """
    The first lyaer of Llama-2-7b-hf with rotation Q applied to the model. 
    Given Q is identity, this model behave the same as the model above.
    """
    def __init__(self, base_model, Q=None, device='cuda'):
        super().__init__()
        self.config = base_model.config
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Ensure components are in float32
        self.embed_tokens = copy.deepcopy(base_model.model.embed_tokens).to(device).float()
        self.first_layer = copy.deepcopy(base_model.model.layers[0]).to(device).float()
        self.rotary_emb = LlamaRotaryEmbedding(self.config).to(device).float()

        # Initialize rotation matrix Q
        if Q is None:
            self.Q = nn.Parameter(torch.eye(self.hidden_size, dtype=torch.float32, device=device))
            self.Q.requires_grad = False
        else:
            # If Q is provided, use it directly without creating a new Parameter
            self.Q = Q  # Q should already be a Parameter with requires_grad=True

        # Freeze all parameters except Q
        for param in self.parameters():
            if param is not self.Q:  # Don't freeze Q
                param.requires_grad = False

    def forward(self, input_ids):
        # Handle input shape: [batch_size, num_chunks, seq_len] -> [batch_size * num_chunks, seq_len]
        if len(input_ids.shape) == 3:
            batch_size, num_chunks, seq_len = input_ids.shape
            input_ids = input_ids.reshape(-1, seq_len)
        else:
            batch_size, seq_len = input_ids.shape

        device = input_ids.device

        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # First rotation after embedding
        hidden_states = hidden_states @ self.Q

        # Position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Rotary embeddings
        dummy_tensor = torch.zeros(
            (batch_size, self.num_heads, seq_len, self.head_dim),
            device=device,
            dtype=hidden_states.dtype
        )
        cos, sin = self.rotary_emb(dummy_tensor, position_ids=position_ids)

        # Get the first layer components
        layer = self.first_layer
        
        # Apply RMSNorm
        hidden_states = layer.input_layernorm(hidden_states)
        
        # Rotate before up and gate projections
        hidden_states = hidden_states @ self.Q.T
        
        # Apply up and gate projections
        up_proj = layer.mlp.up_proj(hidden_states)
        gate_proj = layer.mlp.gate_proj(hidden_states)
        
        # Apply activation and down projection
        hidden_states = layer.mlp.act_fn(gate_proj) * up_proj
        hidden_states = layer.mlp.down_proj(hidden_states)
        
        # Final rotation after down projection
        hidden_states = hidden_states @ self.Q

        return hidden_states


class SequenceChunksDataset(Dataset):
    def __init__(self, tensor, chunk_size):
        # Keep the batch dimension
        self.tensor = tensor  # Remove squeeze(0)
        self.chunk_size = chunk_size
        self.num_chunks = len(self.tensor[0]) // chunk_size  # Use first batch dimension

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        # Return with batch dimension
        return self.tensor[:, start:end]

def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
        
    if eval_mode:
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def custom_loss(xnew, xold, threshold, power):
    # Ensure inputs are float32 and require gradients
    xnew = xnew.float()
    xold = xold.float()
    
    # Compute statistics for new activations
    mean_new = torch.mean(xnew)
    std_new = torch.std(xnew)
    outlier_scores_new = ((torch.abs(xnew - mean_new) / (std_new + 1e-6))/threshold)**power
    
    # Compute statistics for old activations (no gradients needed)
    with torch.no_grad():
        mean_old = torch.mean(xold)
        std_old = torch.std(xold)
        outlier_scores_old = ((torch.abs(xold - mean_old) / (std_old + 1e-6))/threshold)**power
    
    # Compute final loss
    loss = (torch.mean(outlier_scores_new) - torch.mean(outlier_scores_old))*10000
    
    return loss

def main(args):
    threshold = args.threshold
    power = args.power
    num_epochs = args.num_epochs
    print(f"Starting training with threshold={threshold}, power={power}, epochs={num_epochs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("saved_tensors", exist_ok=True)

    trainloader = get_wikitext2(128, 42, 2048, "meta-llama/Llama-2-7b-hf", None, True)
    input_ids = trainloader["input_ids"]
    chunk_size = 2048
    input_dataset = SequenceChunksDataset(input_ids, chunk_size)

    batch_size = 32 
    dataloader = DataLoader(input_dataset, batch_size=batch_size, shuffle=True)

    # Initialize Q as a parameter with gradients enabled
    hadamard_matrix = hadamard_utils.random_hadamard_matrix(4096, "cuda").float()
    Q = nn.Parameter(hadamard_matrix.clone(), requires_grad=True)

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float32, device_map="cpu")
    
    # Original model (no rotation)
    original = BaseLLaMAFirstLayer(model, device="cuda")
    original.eval()

    # Rotated model with Q
    rotated = RotatedLLaMAFirstLayer(model, Q=Q, device="cuda")

    torch.compile(original)
    torch.compile(rotated)
    optimizer = SGDG([rotated.Q], lr=0.01, stiefel=True)
    
    num_batches = len(input_ids)//batch_size+1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_logs_{timestamp}.json"
    losses = []

    def verify_orthogonality(matrix):
        Q = matrix.t() @ matrix
        error = torch.norm(Q - torch.eye(matrix.size(0), device=matrix.device))
        return Q, error

    initial_Q, initial_error = verify_orthogonality(Q.detach())
    print(f"Initial orthogonality error: {initial_error.item():.6f}")
    
    for epoch in range(num_epochs):
        ct = 0
        total_loss = 0.0
        epoch_losses = []
        
        for batch in dataloader:
            ct += 1
            x = batch.to(device, non_blocking=True)  # Remove unsqueeze(0)
            
            # Get original output (no rotation)
            with torch.no_grad():
                orig_output = original(x)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with gradient computation
            rot_output = rotated(x)
            
            # Compute loss
            loss = custom_loss(xnew=rot_output, xold=orig_output, threshold=threshold, power=power)
            
            if not loss.requires_grad:
                continue
                
            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_losses.append(loss.item())

            if ct % 100 == 0:
                print(f"Batch {ct}, Loss: {loss.item():.6f}")

        avg_epoch_loss = total_loss / ct
        losses.append({
            'epoch': epoch,
            'average_loss': avg_epoch_loss,
            'batch_losses': epoch_losses
        })
        
        print(f"\nEpoch {epoch} completed:")
        print(f"Average loss: {avg_epoch_loss:.6f}")

        I, ortho_error = verify_orthogonality(rotated.Q.detach())
        print(f"Orthogonality error: {ortho_error.item():.6f}")

        matrix_path = f"saved_tensors/orthogonal_matrix_epoch_{epoch}_T{threshold}_P{power}_B{batch_size}.pt"
        torch.save(rotated.Q.detach(), matrix_path)
        print(f"Matrix saved to {matrix_path}")
        
        with open(log_file, 'w') as f:
            json.dump(losses, f, indent=4)
        print(f"Losses logged to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--power", type=float, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()
    main(args)




