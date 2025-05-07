import torch # Import PyTorch

def make_skew_hook(layer_name, stats_store):
    """
    Creates a forward hook function for a specific layer to compute skew and stats.
    
    Args:
        layer_name (str): Name of the layer for which the hook is registered.
        stats_store (dict): Dictionary where computed statistics will be stored.

    Returns:
        hook (function): The forward hook function.
    """
    def hook(module, input, output):
        # Detach the output from the computation graph, move to CPU, and flatten to 1D tensor
        flat = output.detach().cpu().flatten()

        # Store layer-wise statistics in the shared dictionary
        stats_store[layer_name] = {
            "mean": flat.mean().item(),  # Mean of the layer output
            "std": flat.std().item(),    # Standard deviation
            "min": flat.min().item(),    # Minimum value
            "max": flat.max().item(),    # Maximum value

            # Skewness = E[(X - mean)^3] / (std^3)
            # Add small epsilon (1e-6) in denominator to avoid division by zero
            "skew": (((flat - flat.mean())**3).mean().item()) / (flat.std().item()**3 + 1e-6)
        }
    return hook  # Return the constructed hook function

def analyze_skew(model, calib_loader, target_layer_names, device="cuda"):
    """
    Attaches hooks to specified layers, runs the model on a calibration batch, 
    and collects output statistics including skew.

    Args:
        model (torch.nn.Module): The neural network model to analyze.
        calib_loader (list or DataLoader): A list or loader that provides input samples.
        target_layer_names (list): Substrings of layer names to match for hooking.
        device (str): Device to run the model on ("cuda" or "cpu").

    Returns:
        stats_store (dict): Dictionary mapping layer names to their output statistics.
    """
    hooks = []        # List to keep track of hook handles so they can be removed later
    stats_store = {}  # Dictionary to store statistics per layer

    # Move model to the target device (usually GPU)
    model.to(device)

    # Iterate through all modules (layers) of the model
    for name, module in model.named_modules():
        # If any of the target layer name substrings match the current layer's name
        if any(t in name for t in target_layer_names):
            # Create a hook for this layer and register it
            hook_fn = make_skew_hook(name, stats_store)
            hooks.append(module.register_forward_hook(hook_fn))

    # Switch model to evaluation mode to disable dropout, etc.
    model.eval()

    # Run a single forward pass to trigger the hooks
    with torch.no_grad():  # Disable gradient tracking for efficiency
        inp, _ = calib_loader[0]  # Take the first batch from the calibration set
        inp = inp.to(device)      # Move input batch to the same device as model
        model(inp)                # Forward pass; hooks collect outputs

    # Remove all hooks after analysis is done
    for h in hooks:
        h.remove()

    return stats_store  # Return the collected statistics

# def decide_quant_mode(stats):
#     """
#     Decide whether to use symmetric or asymmetric quantization
#     based on skewness and mean/std ratio.

#     Args:
#         stats (dict): Dictionary with 'mean', 'std', and 'skew' of a layer.

#     Returns:
#         str: 'symmetric' or 'asymmetric'
#     """
#     mean = stats["mean"]
#     std = stats["std"]
#     skew = stats["skew"]

#     # Heuristic thresholds (can be tuned)
#     if abs(mean) / (std + 1e-6) > 0.5 or abs(skew) > 1.0:
#         return "asymmetric"
#     return "symmetric"

def decide_quant_mode(stats, mean_std_thresh, skew_thresh):
    mean = stats["mean"]
    std = stats["std"]
    skew = stats["skew"]

    if abs(mean) / (std + 1e-6) > mean_std_thresh or abs(skew) > skew_thresh:
        return "asymmetric"
    return "symmetric"
    
def determine_layer_symmetry(layer_name, skew_stats, mean_std_thresh, skew_thresh):
    if layer_name in skew_stats:
        mode = decide_quant_mode(skew_stats[layer_name], mean_std_thresh, skew_thresh)
        print(f"[Auto Quant Mode] {layer_name}: skew = {skew_stats[layer_name]['skew']:.3f} → using {mode}")
        return (mode == "symmetric")
    else:
        return True  # default to symmetric

