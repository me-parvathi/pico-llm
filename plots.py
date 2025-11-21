import matplotlib.pyplot as plt
import numpy as np


def plot_batch_losses(loss_dict, save_path=None):
    """
    Plot training loss per batch for each model.
    
    Args:
        loss_dict: Dictionary mapping model names to lists of loss values
                  Format: {model_name: [loss1, loss2, ...]}
        save_path: Optional path to save the figure. If None, figure is not saved.
    """
    plt.figure(figsize=(10, 5))
    
    for model_name, losses in loss_dict.items():
        plt.plot(losses, label=model_name)
    
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.close()


def plot_epoch_losses(epoch_loss_dict, save_path=None):
    """
    epoch_loss_dict: dict { model_name: [epoch_loss1, epoch_loss2, ...] }
    """
    plt.figure(figsize=(10, 5))
    
    for model_name, losses in epoch_loss_dict.items():
        losses_array = np.array(losses)
        if len(losses_array) > 0:
            # Compute moving average with window=3
            window = 3
            if len(losses_array) >= window:
                # Use np.convolve for smoothing
                kernel = np.ones(window) / window
                smoothed = np.convolve(losses_array, kernel, mode='valid')
                # Pad the beginning to match original length
                smoothed = np.concatenate([losses_array[:window-1], smoothed])
            else:
                smoothed = losses_array
            
            plt.plot(smoothed, label=model_name)
    
    plt.xlabel('Epoch Index')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch (Moving Average, window=3)')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.close()


def plot_nucleus_distribution(sorted_probs, p=0.95, top_k=50, save_path=None):
    """
    sorted_probs: 1D torch tensor sorted in descending order.
    """
    # Convert to numpy, take top_k
    probs_np = sorted_probs[:top_k].detach().cpu().numpy()
    
    # Compute cumulative sum
    cumsum_probs = np.cumsum(probs_np)
    
    # Find cutoff index using numpy searchsorted
    cutoff_idx = np.searchsorted(cumsum_probs, p, side='right')
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Subplot 1: Bar plot of sorted probabilities
    ax1.bar(range(len(probs_np)), probs_np)
    ax1.set_xlabel('Token Index (sorted)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Sorted Probabilities (Top-K)')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Line plot of cumulative sum with vertical line at p cutoff
    ax2.plot(cumsum_probs, label='Cumulative Sum')
    ax2.axvline(x=cutoff_idx, color='r', linestyle='--', label=f'Cutoff at p={p}')
    ax2.axhline(y=p, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Token Index (sorted)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Probability Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.close()


def plot_attention_heatmap(attn_matrix, save_path=None):
    """
    attn_matrix: numpy array [seq_len, seq_len] for a single head.
    """
    seq_len = attn_matrix.shape[0]
    
    # Create figure with better size
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use 'hot' colormap for better attention visualization
    # 'hot' goes from black -> red -> yellow -> white (like heat)
    im = ax.imshow(attn_matrix, cmap="hot", aspect="auto", 
                   vmin=0, vmax=1, origin='upper')
    
    # Add title
    ax.set_title("Attention Weights (Head 0)", fontsize=14, fontweight='bold')
    
    # Set ticks to show actual position indices
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(range(seq_len))
    ax.set_yticklabels(range(seq_len))
    
    # Add labels with better formatting
    ax.set_xlabel("Key Position", fontsize=12)
    ax.set_ylabel("Query Position", fontsize=12)
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, label="Attention Weight")
    cbar.ax.tick_params(labelsize=10)
    
    # Add gridlines to separate cells
    ax.set_xticks([x - 0.5 for x in range(1, seq_len)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, seq_len)], minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    # Annotate cells with actual values
    for i in range(seq_len):
        for j in range(seq_len):
            text = ax.text(j, i, f'{attn_matrix[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if attn_matrix[i, j] > 0.5 else "black",
                          fontsize=10)
    
    # Save with higher DPI for better quality
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_multihead_attention(attn_tensor, save_path=None, max_heads=None):
    """
    Visualize all attention heads in a grid layout.
    
    Args:
        attn_tensor: numpy array [n_heads, seq_len, seq_len] for all heads in a layer.
        save_path: Optional path to save the figure.
        max_heads: Optional maximum number of heads to display.
    """
    n_heads, seq_len, _ = attn_tensor.shape
    
    if max_heads is not None:
        n_heads = min(n_heads, max_heads)
        attn_tensor = attn_tensor[:max_heads]
    
    # Calculate grid size (prefer wider grids)
    cols = min(4, n_heads)  # Max 4 columns
    rows = (n_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Find global min/max for consistent color scaling
    vmin, vmax = attn_tensor.min(), attn_tensor.max()
    
    for head_idx in range(n_heads):
        ax = axes[head_idx]
        im = ax.imshow(attn_tensor[head_idx], cmap="hot", aspect="auto",
                      vmin=vmin, vmax=vmax, origin='upper')
        
        ax.set_title(f"Head {head_idx}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Key Position", fontsize=9)
        ax.set_ylabel("Query Position", fontsize=9)
        
        # Set integer ticks
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        
        # Add gridlines
        ax.set_xticks([x - 0.5 for x in range(1, seq_len)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, seq_len)], minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.3, alpha=0.5)
        
        # Annotate with values (smaller font for multi-head view)
        for i in range(seq_len):
            for j in range(seq_len):
                val = attn_tensor[head_idx, i, j]
                ax.text(j, i, f'{val:.2f}',
                       ha="center", va="center",
                       color="white" if val > 0.5 else "black",
                       fontsize=7)
    
    # Hide unused subplots
    for idx in range(n_heads, len(axes)):
        axes[idx].axis('off')
    
    # Add shared colorbar
    fig.colorbar(im, ax=axes, label="Attention Weight", 
                 orientation='horizontal', pad=0.05, aspect=40)
    
    fig.suptitle("Multi-Head Attention Weights", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_attention_comparison(attn_tensor, tokens=None, save_path=None):
    """
    Show statistical analysis of attention patterns across heads.
    
    Args:
        attn_tensor: numpy array [n_heads, seq_len, seq_len]
        tokens: Optional list of token strings for labeling
        save_path: Optional path to save the figure
    """
    n_heads, seq_len, _ = attn_tensor.shape
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Average attention across all heads
    mean_attn = attn_tensor.mean(axis=0)
    im1 = axes[0].imshow(mean_attn, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Mean Attention\n(Averaged Across All Heads)", fontweight='bold')
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")
    plt.colorbar(im1, ax=axes[0])
    
    # Add gridlines
    axes[0].set_xticks([x - 0.5 for x in range(1, seq_len)], minor=True)
    axes[0].set_yticks([y - 0.5 for y in range(1, seq_len)], minor=True)
    axes[0].grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    # 2. Std deviation - shows head specialization
    std_attn = attn_tensor.std(axis=0)
    im2 = axes[1].imshow(std_attn, cmap="viridis", vmin=0)
    axes[1].set_title("Attention Std Dev\n(Head Specialization)", fontweight='bold')
    axes[1].set_xlabel("Key Position")
    axes[1].set_ylabel("Query Position")
    plt.colorbar(im2, ax=axes[1])
    
    # Add gridlines
    axes[1].set_xticks([x - 0.5 for x in range(1, seq_len)], minor=True)
    axes[1].set_yticks([y - 0.5 for y in range(1, seq_len)], minor=True)
    axes[1].grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    # 3. Max attention (most attended position per head)
    max_attn = attn_tensor.max(axis=0)
    im3 = axes[2].imshow(max_attn, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Max Attention\n(Across All Heads)", fontweight='bold')
    axes[2].set_xlabel("Key Position")
    axes[2].set_ylabel("Query Position")
    plt.colorbar(im3, ax=axes[2])
    
    # Add gridlines
    axes[2].set_xticks([x - 0.5 for x in range(1, seq_len)], minor=True)
    axes[2].set_yticks([y - 0.5 for y in range(1, seq_len)], minor=True)
    axes[2].grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    # Add token labels if provided
    if tokens is not None:
        for ax in axes:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_flow(attn_matrix, tokens=None, threshold=0.1, save_path=None):
    """
    Visualize attention as a flow diagram showing strong connections.
    
    Args:
        attn_matrix: numpy array [seq_len, seq_len] for a single head
        tokens: Optional list of token strings for labeling
        threshold: Only show attention weights above this value
        save_path: Optional path to save the figure
    """
    seq_len = attn_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=(max(10, seq_len * 2), 8))
    
    # Calculate positions for nodes
    spacing = 2.0
    key_positions = np.arange(seq_len) * spacing
    query_positions = np.arange(seq_len) * spacing
    
    # Draw connections (arrows from key to query)
    for q_idx in range(seq_len):
        for k_idx in range(seq_len):
            weight = attn_matrix[q_idx, k_idx]
            if weight > threshold:
                # Calculate color based on weight
                color = plt.cm.hot(weight)
                
                # Draw arrow from key (bottom) to query (top)
                ax.annotate('', xy=(q_idx * spacing, 2.0), 
                           xytext=(k_idx * spacing, 0.0),
                           arrowprops=dict(arrowstyle='->', lw=weight*3, 
                                         color=color, alpha=min(weight + 0.3, 1.0)))
                
                # Add weight label at midpoint
                mid_x = (k_idx * spacing + q_idx * spacing) / 2
                ax.text(mid_x, 1.0, f'{weight:.2f}',
                       fontsize=7, alpha=0.7, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.7, edgecolor='none'))
    
    # Draw nodes for keys (bottom)
    ax.scatter(key_positions, [0]*seq_len, s=800, c='lightblue', 
              edgecolors='black', zorder=5, linewidths=2, label='Key Positions')
    
    # Draw nodes for queries (top)
    ax.scatter(query_positions, [2]*seq_len, s=800, c='lightcoral',
              edgecolors='black', zorder=5, linewidths=2, label='Query Positions')
    
    # Add position/token labels
    if tokens is not None:
        for i, token in enumerate(tokens):
            # Key labels (bottom)
            ax.text(i * spacing, -0.4, f'{i}: {token}', ha='center', 
                   fontsize=9, fontweight='bold')
            # Query labels (top)
            ax.text(i * spacing, 2.4, f'{i}: {token}', ha='center', 
                   fontsize=9, fontweight='bold')
    else:
        for i in range(seq_len):
            # Key labels (bottom)
            ax.text(i * spacing, -0.4, f'{i}', ha='center', 
                   fontsize=10, fontweight='bold')
            # Query labels (top)
            ax.text(i * spacing, 2.4, f'{i}', ha='center', 
                   fontsize=10, fontweight='bold')
    
    ax.set_xlim(-1, seq_len * spacing)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f"Attention Flow Diagram (threshold={threshold})", 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_layerwise_attention(layer_attns, layer_names=None, save_path=None):
    """
    Show attention patterns across different transformer layers.
    
    Args:
        layer_attns: list of [n_heads, seq_len, seq_len] arrays, one per layer
        layer_names: Optional list of layer names/numbers
        save_path: Optional path to save the figure
    """
    n_layers = len(layer_attns)
    
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    
    for layer_idx, (attn, ax) in enumerate(zip(layer_attns, axes)):
        # Average across heads
        mean_attn = attn.mean(axis=0)
        im = ax.imshow(mean_attn, cmap="hot", vmin=0, vmax=1)
        
        layer_name = f"Layer {layer_idx}" if layer_names is None else layer_names[layer_idx]
        ax.set_title(f"{layer_name}\n(Mean Attention)", fontweight='bold')
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        
        # Add gridlines
        seq_len = attn.shape[1]
        ax.set_xticks([x - 0.5 for x in range(1, seq_len)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, seq_len)], minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        
        plt.colorbar(im, ax=ax)
    
    fig.suptitle("Attention Patterns Across Layers", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

