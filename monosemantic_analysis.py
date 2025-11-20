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
                        - "mlp_post": post-SiLU activations (seq_len, batch, d_model * 4)
    
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
                mlp_post = batch_activations["layers"][layer_idx]["mlp_post"]
                # mlp_post shape: (seq_len, batch, d_model * 4)
                # Flatten to (total_tokens, num_neurons)
                seq_len, batch_size, num_neurons = mlp_post.shape
                mlp_post_flat = mlp_post.view(-1, num_neurons)  # (seq_len * batch, num_neurons)
                layer_activations_list.append(mlp_post_flat)
        
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


def build_sequence_index_map(activations):
    """
    Build a mapping from global sequence index to (batch_idx, batch_item).
    
    Args:
        activations: Dictionary loaded from activation_recorder.py with keys:
            - "activations": list of activation dicts (one per batch)
    
    Returns:
        list: Mapping where sequence_map[global_seq_idx] = (batch_idx, batch_item)
    """
    sequence_map = []
    all_batch_activations = activations["activations"]
    
    for batch_idx, batch_activations in enumerate(all_batch_activations):
        # Get batch_size from layers if available, otherwise from embeddings
        batch_size = None
        
        if len(batch_activations.get("layers", [])) > 0:
            # Use first layer to get batch_size
            first_layer_mlp = batch_activations["layers"][0]["mlp_post"]
            batch_size = first_layer_mlp.shape[1]  # batch dimension
        elif "embeddings" in batch_activations:
            # Use embeddings to infer batch_size
            embeddings = batch_activations["embeddings"]
            if isinstance(embeddings, torch.Tensor) and embeddings.dim() >= 2:
                batch_size = embeddings.shape[1]
        
        if batch_size is None:
            continue
        
        # Add mapping for each item in this batch
        for batch_item in range(batch_size):
            sequence_map.append((batch_idx, batch_item))
    
    return sequence_map


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
        list: List of tuples (token_string, activation_value, token_id, batch_idx, batch_item, seq_pos)
              where batch_idx is the batch index, batch_item is the sequence index within that batch,
              and seq_pos is the position within the sequence.
              Sorted by activation magnitude (highest first)
    """
    results = []
    
    all_batch_activations = activations["activations"]
    all_token_ids = activations["token_ids"]
    
    # Build sequence index mapping and reverse lookup for efficiency
    sequence_map = build_sequence_index_map(activations)
    reverse_map = {(batch_idx, batch_item): global_idx for global_idx, (batch_idx, batch_item) in enumerate(sequence_map)} if sequence_map else None
    
    # Process each batch
    for batch_idx, batch_activations in enumerate(all_batch_activations):
        if layer >= len(batch_activations["layers"]):
            continue
        
        mlp_post = batch_activations["layers"][layer]["mlp_post"]
        # mlp_post shape: (seq_len, batch, num_neurons)
        seq_len, batch_size, num_neurons = mlp_post.shape
        
        if neuron >= num_neurons:
            continue
        
        # Get activations for this specific neuron
        neuron_activations = mlp_post[:, :, neuron]  # (seq_len, batch)
        
        # Process each position in sequence and each item in batch
        for seq_pos in range(seq_len):
            for batch_item in range(batch_size):
                activation_val = neuron_activations[seq_pos, batch_item].item()
                
                # Map to corresponding token ID using sequence_map
                if reverse_map and (batch_idx, batch_item) in reverse_map:
                    global_seq_idx = reverse_map[(batch_idx, batch_item)]
                else:
                    # Fallback: compute cumulative if mapping doesn't contain this pair
                    cumulative_sequences = 0
                    for i in range(batch_idx):
                        if i < len(all_batch_activations) and len(all_batch_activations[i].get("layers", [])) > 0:
                            prev_mlp_post = all_batch_activations[i]["layers"][0]["mlp_post"]
                            cumulative_sequences += prev_mlp_post.shape[1]
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
                    
                    results.append((token_string, activation_val, token_id, batch_idx, batch_item, seq_pos))
    
    # Sort by absolute activation value (descending)
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Return top k
    return results[:k]


def visualize_neuron_distribution(layer, neuron, activations, by_position=False):
    """
    Plot a histogram of activation values for a specific neuron.
    
    Args:
        layer: Layer index (0-indexed)
        neuron: Neuron index within the layer (0-indexed)
        activations: Dictionary loaded from activation_recorder.py
        by_position: If True, generate histograms per token position (default: False)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if not by_position:
        # Original behavior: single histogram
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
    else:
        # New behavior: histograms per position
        all_batch_activations = activations["activations"]
        
        # Find maximum sequence length
        max_seq_len = 0
        for batch_activations in all_batch_activations:
            if layer < len(batch_activations.get("layers", [])):
                mlp_post = batch_activations["layers"][layer]["mlp_post"]
                seq_len = mlp_post.shape[0]
                max_seq_len = max(max_seq_len, seq_len)
        
        # Limit to first 32 positions
        num_positions = min(max_seq_len, 32)
        
        # Collect activations per position
        position_activations = [[] for _ in range(num_positions)]
        
        for batch_activations in all_batch_activations:
            if layer >= len(batch_activations.get("layers", [])):
                continue
            
            mlp_post = batch_activations["layers"][layer]["mlp_post"]
            # mlp_post shape: (seq_len, batch, num_neurons)
            seq_len, batch_size, num_neurons = mlp_post.shape
            
            if neuron >= num_neurons:
                continue
            
            # Get activations for this specific neuron
            neuron_activations = mlp_post[:, :, neuron]  # (seq_len, batch)
            
            # Collect activations for each position
            for pos in range(min(seq_len, num_positions)):
                pos_activations = neuron_activations[pos, :].cpu().numpy()
                position_activations[pos].extend(pos_activations.tolist())
        
        # Create grid of histograms
        # Calculate grid dimensions (aim for roughly square grid)
        n_cols = int(np.ceil(np.sqrt(num_positions)))
        n_rows = int(np.ceil(num_positions / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if num_positions == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for pos in range(num_positions):
            ax = axes[pos]
            pos_vals = position_activations[pos]
            
            if len(pos_vals) > 0:
                ax.hist(pos_vals, bins=30, edgecolor='black', alpha=0.7)
                ax.set_title(f'Pos {pos}', fontsize=10)
                ax.set_xlabel('Activation', fontsize=8)
                ax.set_ylabel('Frequency', fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Pos {pos}', fontsize=10)
        
        # Hide unused subplots
        for pos in range(num_positions, len(axes)):
            axes[pos].axis('off')
        
        fig.suptitle(f'Activation Distribution by Position: Layer {layer}, Neuron {neuron}', 
                     fontsize=14, y=0.995)
        plt.tight_layout()
        return fig


def get_attention_for_sequence(activations, layer, batch_idx, batch_item):
    """
    Extract attention weights and tokens for a specific sequence.
    
    Args:
        activations: Dictionary loaded from activation_recorder.py
        layer: Layer index (0-indexed)
        batch_idx: Batch index
        batch_item: Sequence index within the batch
    
    Returns:
        tuple: (attention_weights, tokens) where:
            - attention_weights: (n_heads, seq_len, seq_len) tensor
            - tokens: List of token ID tensors for the sequence
    """
    all_batch_activations = activations["activations"]
    all_token_ids = activations["token_ids"]
    
    if batch_idx >= len(all_batch_activations):
        raise ValueError(f"Batch index {batch_idx} out of range")
    
    batch_activations = all_batch_activations[batch_idx]
    if layer >= len(batch_activations["layers"]):
        raise ValueError(f"Layer {layer} not found in batch {batch_idx}")
    
    attention_weights = batch_activations["layers"][layer]["attention"]
    # attention_weights shape: (batch, n_heads, seq_len, seq_len)
    
    if batch_item >= attention_weights.shape[0]:
        raise ValueError(f"Batch item {batch_item} out of range (batch_size={attention_weights.shape[0]})")
    
    # Extract attention for this specific sequence
    seq_attention = attention_weights[batch_item]  # (n_heads, seq_len, seq_len)
    
    # Get tokens for this sequence using sequence_map
    sequence_map = build_sequence_index_map(activations)
    if sequence_map:
        # Build reverse lookup for efficiency
        reverse_map = {(b_idx, b_item): global_idx for global_idx, (b_idx, b_item) in enumerate(sequence_map)}
        if (batch_idx, batch_item) in reverse_map:
            global_seq_idx = reverse_map[(batch_idx, batch_item)]
        else:
            # Fallback: compute cumulative if mapping doesn't contain this pair
            cumulative_sequences = 0
            for i in range(batch_idx):
                if i < len(all_batch_activations) and len(all_batch_activations[i].get("layers", [])) > 0:
                    prev_mlp_post = all_batch_activations[i]["layers"][0]["mlp_post"]
                    cumulative_sequences += prev_mlp_post.shape[1]
            global_seq_idx = cumulative_sequences + batch_item
    else:
        # Fallback: compute cumulative if mapping is empty
        cumulative_sequences = 0
        for i in range(batch_idx):
            if i < len(all_batch_activations) and len(all_batch_activations[i].get("layers", [])) > 0:
                prev_mlp_post = all_batch_activations[i]["layers"][0]["mlp_post"]
                cumulative_sequences += prev_mlp_post.shape[1]
        global_seq_idx = cumulative_sequences + batch_item
    if global_seq_idx >= len(all_token_ids):
        raise ValueError(f"Global sequence index {global_seq_idx} out of range")
    
    seq_tokens = all_token_ids[global_seq_idx]
    
    return seq_attention, seq_tokens


def visualize_attention_weights(attention_tensor, tokens, head_idx=None, title_prefix=""):
    """
    Plot a heatmap of attention weights using matplotlib.
    
    Args:
        attention_tensor: Attention weights tensor. Can be:
            - (batch, n_heads, seq_len, seq_len) - full multi-head attention
            - (n_heads, seq_len, seq_len) - multi-head for single sequence
            - (seq_len, seq_len) - single head, single batch
        tokens: List of token strings or token IDs corresponding to the sequence positions
        head_idx: If attention_tensor has multiple heads, which head to visualize (None = average)
        title_prefix: Optional prefix for the plot title
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Handle different tensor shapes
    if attention_tensor.dim() == 4:
        # (batch, n_heads, seq_len, seq_len) - take first batch
        attention_tensor = attention_tensor[0]  # (n_heads, seq_len, seq_len)
    
    if attention_tensor.dim() == 3:
        # (n_heads, seq_len, seq_len) - either select head or average
        if head_idx is not None:
            attention_tensor = attention_tensor[head_idx]  # (seq_len, seq_len)
            head_label = f" (Head {head_idx})"
        else:
            attention_tensor = attention_tensor.mean(dim=0)  # (seq_len, seq_len)
            head_label = " (Averaged over heads)"
    elif attention_tensor.dim() == 2:
        # (seq_len, seq_len) - already in correct shape
        head_label = ""
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
    title = f'Attention Weights Heatmap{head_label}'
    if title_prefix:
        title = f"{title_prefix}: {title}"
    ax.set_title(title, fontsize=14)
    
    # Add token labels if provided
    if tokens is not None and len(tokens) == attention_matrix.shape[0]:
        # Handle token IDs (integers) vs token strings
        if isinstance(tokens[0], (int, torch.Tensor)):
            # These are token IDs - we'll just show them as numbers for now
            # (decoding should be done before calling this function)
            display_tokens = [str(int(t)) if isinstance(t, torch.Tensor) else str(t) for t in tokens]
        else:
            # These are token strings
            display_tokens = [t[:10] + '...' if len(t) > 10 else t for t in tokens]
        
        ax.set_xticks(range(len(display_tokens)))
        ax.set_yticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(display_tokens, fontsize=8)
    
    plt.tight_layout()
    return fig


def visualize_token_activation_heatmap(layer, neuron, activations):
    """
    Plot a heatmap of neuron activation values across all sequences and token positions.
    
    Args:
        layer: Layer index (0-indexed)
        neuron: Neuron index within the layer (0-indexed)
        activations: Dictionary loaded from activation_recorder.py with keys:
            - "token_ids": list of token id tensors
            - "activations": list of activation dicts (one per batch)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    all_batch_activations = activations["activations"]
    all_token_ids = activations["token_ids"]
    
    # Find maximum sequence length across all sequences
    max_seq_len = 0
    for token_seq in all_token_ids:
        if isinstance(token_seq, torch.Tensor):
            seq_len = len(token_seq)
        else:
            seq_len = len(token_seq)
        max_seq_len = max(max_seq_len, seq_len)
    
    # Count total number of sequences
    num_sequences = len(all_token_ids)
    
    # Initialize matrix with NaN (for masking shorter sequences)
    activation_matrix = np.full((num_sequences, max_seq_len), np.nan)
    
    # Fill matrix with activation values
    cumulative_sequences = 0
    for batch_idx, batch_activations in enumerate(all_batch_activations):
        # Check if we can process this batch
        if layer >= len(batch_activations["layers"]):
            # Can't process this batch - try to infer batch_size to maintain alignment
            # with all_token_ids (which includes sequences from all batches)
            batch_size = None
            if len(batch_activations["layers"]) > 0:
                # Use first available layer to get batch_size
                first_layer_mlp = batch_activations["layers"][0]["mlp_post"]
                batch_size = first_layer_mlp.shape[1]
            elif "embeddings" in batch_activations:
                # Use embeddings to infer batch_size
                embeddings = batch_activations["embeddings"]
                if isinstance(embeddings, torch.Tensor) and embeddings.dim() >= 2:
                    batch_size = embeddings.shape[1]
            
            # If we can determine batch_size, increment to account for sequences in all_token_ids
            # If we can't, we have a problem - skip without incrementing (will cause misalignment)
            if batch_size is not None:
                cumulative_sequences += batch_size
            continue
        
        mlp_post = batch_activations["layers"][layer]["mlp_post"]
        # mlp_post shape: (seq_len, batch, num_neurons)
        seq_len, batch_size, num_neurons = mlp_post.shape
        
        if neuron >= num_neurons:
            # Can't process this batch - but we know batch_size, so increment to maintain alignment
            # with all_token_ids (which includes sequences from all batches)
            cumulative_sequences += batch_size
            continue
        
        # Get activations for this specific neuron
        neuron_activations = mlp_post[:, :, neuron]  # (seq_len, batch)
        
        # Fill in activation values for each sequence in this batch
        for batch_item in range(batch_size):
            global_seq_idx = cumulative_sequences + batch_item
            if global_seq_idx >= num_sequences:
                break
            
            # Get actual sequence length for this sequence
            token_seq = all_token_ids[global_seq_idx]
            if isinstance(token_seq, torch.Tensor):
                actual_seq_len = len(token_seq)
            else:
                actual_seq_len = len(token_seq)
            
            # Extract activations for this sequence (up to actual_seq_len)
            actual_seq_len = min(actual_seq_len, seq_len)
            for pos in range(actual_seq_len):
                activation_matrix[global_seq_idx, pos] = neuron_activations[pos, batch_item].item()
        
        # Only increment after successfully processing the batch
        cumulative_sequences += batch_size
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, num_sequences * 0.3)))
    
    # Plot heatmap using imshow with magma colormap
    im = ax.imshow(activation_matrix, cmap='magma', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Activation Value')
    
    # Set labels
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Sequence Index', fontsize=12)
    ax.set_title(f'Token Activation Heatmap: Layer {layer}, Neuron {neuron}', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_topk_tokens(layer, neuron, top_tokens, save_path):
    """
    Plot horizontal bar chart of top-k tokens and their activation values.
    
    Args:
        layer: Layer index (0-indexed)
        neuron: Neuron index within the layer (0-indexed)
        top_tokens: List of tuples from top_k_tokens_for_neuron, each containing:
            (token_string, activation_value, token_id, batch_idx, batch_item, seq_pos)
        save_path: Path to save the PNG file
    """
    # Extract token_string and activation_val, sort by absolute activation value
    token_data = []
    for entry in top_tokens:
        token_string = entry[0]
        activation_val = entry[1]
        token_data.append((token_string, activation_val))
    
    # Sort by absolute activation value (descending)
    token_data.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Extract sorted tokens and activations
    tokens = [t[0] for t in token_data]
    activations = [t[1] for t in token_data]
    
    # Truncate token strings to max ~12 characters
    truncated_tokens = [t[:12] if len(t) <= 12 else t[:9] + "..." for t in tokens]
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(tokens) * 0.4)))
    
    # Plot horizontal bars
    y_pos = range(len(truncated_tokens))
    ax.barh(y_pos, activations, alpha=0.7)
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(truncated_tokens)
    ax.set_xlabel('Activation Value', fontsize=12)
    ax.set_ylabel('Token', fontsize=12)
    ax.set_title(f'Top-K Tokens: Layer {layer}, Neuron {neuron}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

