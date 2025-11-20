"""
Monosemanticity analysis for pico-llm activations.

This module implements core monosemanticity analysis similar to Anthropic's blog post,
but simplified for the pico-llm architecture.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def compute_neuron_stats(activations):
    """
    Extract all MLP pre-activations across layers and compute statistics for each neuron.
    
    Args:
        activations: Dictionary loaded from activation_recorder.py with keys:
            - "token_ids": list of token id tensors
            - "raw_text": list of raw text strings
            - "activations": list of activation dicts (one per batch)
                Each activation dict has:
                    - "embeddings": token embeddings
                    - "layers": list of layer dicts, each with:
                        - "attention": attention weights
                        - "mlp_pre": pre-MLP activations (seq_len, batch, d_model * 4)
    
    Returns:
        dict: Nested dictionary with structure:
            {
                layer_idx: {
                    neuron_idx: {
                        "mean": float,
                        "std": float,
                        "max": float,
                        "all_activations": torch.Tensor  # All activation values across all tokens
                    }
                }
            }
    """
    stats = {}
    
    # Extract all MLP pre-activations across all batches
    all_batch_activations = activations["activations"]
    
    # Process each layer
    if len(all_batch_activations) == 0:
        return stats
    
    # Get number of layers from first batch
    num_layers = len(all_batch_activations[0]["layers"])
    
    for layer_idx in range(num_layers):
        stats[layer_idx] = {}
        
        # Collect all activations for this layer across all batches
        layer_activations_list = []
        
        for batch_activations in all_batch_activations:
            if layer_idx < len(batch_activations["layers"]):
                mlp_pre = batch_activations["layers"][layer_idx]["mlp_pre"]
                # mlp_pre shape: (seq_len, batch, d_model * 4)
                # Flatten to (total_tokens, num_neurons)
                seq_len, batch_size, num_neurons = mlp_pre.shape
                mlp_pre_flat = mlp_pre.view(-1, num_neurons)  # (seq_len * batch, num_neurons)
                layer_activations_list.append(mlp_pre_flat)
        
        if len(layer_activations_list) == 0:
            continue
        
        # Concatenate all batches
        all_layer_activations = torch.cat(layer_activations_list, dim=0)  # (total_tokens, num_neurons)
        
        # Compute stats for each neuron
        num_neurons = all_layer_activations.shape[1]
        for neuron_idx in range(num_neurons):
            neuron_activations = all_layer_activations[:, neuron_idx]  # (total_tokens,)
            
            stats[layer_idx][neuron_idx] = {
                "mean": neuron_activations.mean().item(),
                "std": neuron_activations.std().item(),
                "max": neuron_activations.max().item(),
                "all_activations": neuron_activations
            }
    
    return stats


def top_k_tokens_for_neuron(layer, neuron, activations, tokenizer, k=20):
    """
    Find the top k tokens that most activate a specific neuron.
    
    Args:
        layer: Layer index (0-indexed)
        neuron: Neuron index within the layer (0-indexed)
        activations: Dictionary loaded from activation_recorder.py
        tokenizer: Tokenizer/encoder object (e.g., tiktoken encoder) with decode method
        k: Number of top tokens to return (default=20)
    
    Returns:
        list: List of tuples (token_string, activation_value, token_id, batch_idx, seq_pos)
              Sorted by activation magnitude (highest first)
    """
    results = []
    
    all_batch_activations = activations["activations"]
    all_token_ids = activations["token_ids"]
    
    # Calculate cumulative sequence count to map batch items to token_ids
    # token_ids is a flat list where sequences from all batches are concatenated
    cumulative_sequences = 0
    
    # Process each batch
    for batch_idx, batch_activations in enumerate(all_batch_activations):
        if layer >= len(batch_activations["layers"]):
            continue
        
        mlp_pre = batch_activations["layers"][layer]["mlp_pre"]
        # mlp_pre shape: (seq_len, batch, num_neurons)
        seq_len, batch_size, num_neurons = mlp_pre.shape
        
        if neuron >= num_neurons:
            # Update cumulative count even if neuron is invalid
            cumulative_sequences += batch_size
            continue
        
        # Get activations for this specific neuron
        neuron_activations = mlp_pre[:, :, neuron]  # (seq_len, batch)
        
        # Process each position in sequence and each item in batch
        for seq_pos in range(seq_len):
            for batch_item in range(batch_size):
                activation_val = neuron_activations[seq_pos, batch_item].item()
                
                # Map to corresponding token ID
                # token_ids is indexed by: cumulative_sequences + batch_item
                global_seq_idx = cumulative_sequences + batch_item
                
                if global_seq_idx < len(all_token_ids):
                    seq_token_ids = all_token_ids[global_seq_idx]
                    # Handle both tensor and list formats
                    if isinstance(seq_token_ids, torch.Tensor):
                        if seq_pos < len(seq_token_ids):
                            token_id = seq_token_ids[seq_pos].item()
                        else:
                            continue
                    else:
                        if seq_pos < len(seq_token_ids):
                            token_id = seq_token_ids[seq_pos]
                        else:
                            continue
                    
                    try:
                        token_string = tokenizer.decode([token_id])
                    except:
                        token_string = f"<token_{token_id}>"
                    
                    results.append((token_string, activation_val, token_id, batch_idx, seq_pos))
        
        # Update cumulative count after processing this batch
        cumulative_sequences += batch_size
    
    # Sort by absolute activation value (descending)
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Return top k
    return results[:k]


def visualize_neuron_distribution(layer, neuron, activations):
    """
    Plot a histogram of activation values for a specific neuron.
    
    Args:
        layer: Layer index (0-indexed)
        neuron: Neuron index within the layer (0-indexed)
        activations: Dictionary loaded from activation_recorder.py
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Get neuron stats to access all activations
    stats = compute_neuron_stats(activations)
    
    if layer not in stats or neuron not in stats[layer]:
        raise ValueError(f"No data found for layer {layer}, neuron {neuron}")
    
    neuron_data = stats[layer][neuron]["all_activations"]
    
    # Convert to numpy for plotting
    activation_values = neuron_data.cpu().numpy()
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(activation_values, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Activation Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Activation Distribution: Layer {layer}, Neuron {neuron}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = stats[layer][neuron]["mean"]
    std_val = stats[layer][neuron]["std"]
    max_val = stats[layer][neuron]["max"]
    stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMax: {max_val:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def visualize_attention_weights(attention_tensor, tokens):
    """
    Plot a heatmap of attention weights using matplotlib.
    
    Args:
        attention_tensor: Attention weights tensor. Can be:
            - (batch, n_heads, seq_len, seq_len) - full multi-head attention
            - (n_heads, seq_len, seq_len) - single batch
            - (seq_len, seq_len) - single head, single batch
        tokens: List of token strings corresponding to the sequence positions
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Handle different tensor shapes
    if attention_tensor.dim() == 4:
        # (batch, n_heads, seq_len, seq_len) - take first batch, average over heads
        attention_tensor = attention_tensor[0].mean(dim=0)  # (seq_len, seq_len)
    elif attention_tensor.dim() == 3:
        # (n_heads, seq_len, seq_len) - average over heads
        attention_tensor = attention_tensor.mean(dim=0)  # (seq_len, seq_len)
    elif attention_tensor.dim() == 2:
        # (seq_len, seq_len) - already in correct shape
        pass
    else:
        raise ValueError(f"Unexpected attention tensor shape: {attention_tensor.shape}")
    
    # Convert to numpy
    attention_matrix = attention_tensor.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap using imshow
    im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    # Set labels
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title('Attention Weights Heatmap', fontsize=14)
    
    # Add token labels if provided
    if tokens is not None and len(tokens) == attention_matrix.shape[0]:
        # Truncate long token strings for display
        display_tokens = [t[:10] + '...' if len(t) > 10 else t for t in tokens]
        ax.set_xticks(range(len(display_tokens)))
        ax.set_yticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(display_tokens, fontsize=8)
    
    plt.tight_layout()
    return fig

