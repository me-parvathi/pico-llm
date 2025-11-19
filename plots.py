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
    plt.figure()
    im = plt.imshow(attn_matrix, cmap="viridis")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.colorbar(im, label="Attention weight")
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.close()

