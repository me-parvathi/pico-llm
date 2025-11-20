"""
Orchestrates the full interpretability pipeline for pico-llm.

This script:
1. Loads a trained model from checkpoint
2. Collects activations from the model
3. Saves and loads activations
4. Runs monosemantic analysis
5. Generates visualizations for selected neurons
"""

import torch
import tiktoken
import importlib.util
import sys
import os
import argparse
import glob
from pathlib import Path
from datasets import load_dataset
from activation_recorder import collect_activations, save_activations, load_activations
from monosemantic_analysis import (
    compute_neuron_stats,
    top_k_tokens_for_neuron,
    visualize_neuron_distribution,
    visualize_token_activation_heatmap,
    plot_topk_tokens
)

# Import pico-llm module (handles hyphenated filename)
# Get the directory where this script is located
script_dir = Path(__file__).parent
pico_llm_path = script_dir / "pico-llm.py"

spec = importlib.util.spec_from_file_location("pico_llm", str(pico_llm_path))
pico_llm = importlib.util.module_from_spec(spec)
sys.modules["pico_llm"] = pico_llm
spec.loader.exec_module(pico_llm)

TransformerModel = pico_llm.TransformerModel
MixedSequenceDataset = pico_llm.MixedSequenceDataset
seq_collate_fn = pico_llm.seq_collate_fn


def load_model(checkpoint_path, device="cpu"):
    """
    Load a trained TransformerModel from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load the model on
    
    Returns:
        Loaded TransformerModel
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint if available, otherwise use defaults
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        vocab_size = config.get('vocab_size', 50257)
        d_model = config.get('d_model', 768)
        n_heads = config.get('n_heads', 12)
        n_blocks = config.get('n_blocks', 6)
        block_size = config.get('block_size', 64)
    else:
        # Defaults (for old checkpoints without config)
        vocab_size = 50257
        d_model = 768
        n_heads = 12
        n_blocks = 6
        block_size = 64
    
    # Create model with architecture from config
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_blocks=n_blocks,
        block_size=block_size
    )
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume it's a state_dict directly
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def setup_tokenizer_and_dataset(block_size=64, train_subset_size=20000, batch_size=16,
                                input_files=None, tinystories_weight=1.0):
    """
    Set up tokenizer and dataset matching pico-llm.py logic.
    
    Args:
        block_size: Maximum sequence length
        train_subset_size: Number of samples from TinyStories
        batch_size: Batch size for DataLoader
        input_files: List of custom text files to use (optional)
        tinystories_weight: Probability of sampling from TinyStories (0.0 to skip)
    
    Returns:
        tuple: (tokenizer, dataloader)
    """
    # Setup tokenizer
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")
    
    tinystories_seqs = []
    other_seqs = []
    
    # Load TinyStories dataset if weight > 0
    if tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
        
        # Tokenize sequences
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
    
    # Load custom input files if provided
    if input_files:
        for filepath in input_files:
            print(f"Reading custom text file: {filepath}")
            if not os.path.exists(filepath):
                print(f"  Warning: File '{filepath}' not found, skipping...")
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")
    
    # Create dataset
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=tinystories_weight
    )
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )
    
    return enc, train_loader


def find_latest_checkpoint(checkpoint_dir="checkpoints", model_name="transformer"):
    """
    Find the latest checkpoint in the checkpoints directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Model name to search for (e.g., "transformer", "lstm_seq")
    
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for epoch checkpoints first (most recent training)
    epoch_pattern = os.path.join(checkpoint_dir, f"{model_name}_epoch_*.pt")
    epoch_checkpoints = glob.glob(epoch_pattern)
    
    if epoch_checkpoints:
        # Sort by epoch number (extract from filename)
        def get_epoch_num(path):
            try:
                # Extract epoch number from filename like "transformer_epoch_3.pt"
                basename = os.path.basename(path)
                epoch_str = basename.split("_epoch_")[1].split(".pt")[0]
                return int(epoch_str)
            except:
                return -1
        
        epoch_checkpoints.sort(key=get_epoch_num, reverse=True)
        return epoch_checkpoints[0]
    
    # Fallback to step checkpoints
    step_pattern = os.path.join(checkpoint_dir, f"{model_name}_step_*.pt")
    step_checkpoints = glob.glob(step_pattern)
    
    if step_checkpoints:
        # Sort by step number
        def get_step_num(path):
            try:
                basename = os.path.basename(path)
                step_str = basename.split("_step_")[1].split(".pt")[0]
                return int(step_str)
            except:
                return -1
        
        step_checkpoints.sort(key=get_step_num, reverse=True)
        return step_checkpoints[0]
    
    return None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run interpretability pipeline for pico-llm")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file. If not specified, will look for latest in checkpoints/ directory.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints (default: 'checkpoints')")
    parser.add_argument("--model_name", type=str, default="transformer",
                        help="Model name to search for in checkpoint directory (default: 'transformer')")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to use for dataset (should match training config)")
    parser.add_argument("--tinystories_weight", type=float, default=1.0,
                        help="Probability of sampling from TinyStories (default: 1.0, set to 0.0 to skip)")
    parser.add_argument("--max_batches", type=int, default=10,
                        help="Maximum number of batches to collect activations from (default: 10)")
    parser.add_argument("--block_size", type=int, default=64,
                        help="Maximum sequence length (default: 64)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for DataLoader (default: 16)")
    parser.add_argument("--device_id", type=str, default=None,
                        help="Torch device identifier (default: auto-detect)")
    parser.add_argument("--positional_histograms", action="store_true",
                        help="Generate histograms per token position instead of single histogram")
    return parser.parse_args()


def main():
    """
    Main interpretability pipeline.
    """
    args = parse_args()
    
    # Configuration
    activations_path = "activations.pt"
    device = args.device_id
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"Requested device '{device}' but CUDA not available. Falling back to CPU.")
            device = "cpu"
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print(f"Looking for latest checkpoint in '{args.checkpoint_dir}' directory...")
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir, args.model_name)
        if checkpoint_path is None:
            print(f"Error: No checkpoint found in '{args.checkpoint_dir}' directory.")
            print(f"Please specify a checkpoint with --checkpoint or ensure checkpoints exist.")
            return
        print(f"Found checkpoint: {checkpoint_path}")
    
    print("=" * 60)
    print("Pico-LLM Interpretability Pipeline")
    print("=" * 60)
    
    # Step 1: Load trained model from checkpoint
    print(f"\n[Step 1] Loading model from {checkpoint_path}...")
    try:
        model = load_model(checkpoint_path, device=device)
        print(f"Model loaded successfully on device: {device}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        print("Please ensure the checkpoint exists before running this script.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Load tokenizer and dataset
    print(f"\n[Step 2] Setting up tokenizer and dataset...")
    print(f"  Configuration: tinystories_weight={args.tinystories_weight}, input_files={args.input_files}")
    tokenizer, dataset = setup_tokenizer_and_dataset(
        block_size=args.block_size,
        batch_size=args.batch_size,
        input_files=args.input_files,
        tinystories_weight=args.tinystories_weight
    )
    
    # Step 3: Collect activations
    print(f"\n[Step 3] Collecting activations (max_batches={args.max_batches})...")
    try:
        activations = collect_activations(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            max_batches=args.max_batches
        )
        print(f"Collected activations from {len(activations['activations'])} batches")
    except Exception as e:
        print(f"Error collecting activations: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Save activations
    print(f"\n[Step 4] Saving activations to {activations_path}...")
    try:
        save_activations(activations_path, activations)
        print("Activations saved successfully")
    except Exception as e:
        print(f"Error saving activations: {e}")
        return
    
    # Step 5: Load activations
    print(f"\n[Step 5] Loading activations from {activations_path}...")
    try:
        loaded_activations = load_activations(activations_path)
        print("Activations loaded successfully")
    except Exception as e:
        print(f"Error loading activations: {e}")
        return
    
    # Step 6: Compute neuron stats
    print(f"\n[Step 6] Computing neuron statistics...")
    try:
        neuron_stats = compute_neuron_stats(loaded_activations)
        print(f"Computed stats for {len(neuron_stats)} layers")
        for layer_idx in neuron_stats:
            print(f"  Layer {layer_idx}: {len(neuron_stats[layer_idx])} neurons")
    except Exception as e:
        print(f"Error computing neuron stats: {e}")
        return
    
    # Step 7: Analyze selected neurons
    print(f"\n[Step 7] Analyzing selected neurons...")
    
    # Select a few interesting neurons to analyze
    # We'll pick neurons from different layers
    selected_neurons = [
        (0, 0),   # Layer 0, Neuron 0
        (0, 100), # Layer 0, Neuron 100
        (2, 0),   # Layer 2, Neuron 0
        (2, 500), # Layer 2, Neuron 500
    ]
    
    # Filter to only valid neurons
    valid_neurons = []
    for layer, neuron in selected_neurons:
        if layer in neuron_stats and neuron in neuron_stats[layer]:
            valid_neurons.append((layer, neuron))
        else:
            print(f"  Warning: Layer {layer}, Neuron {neuron} not found, skipping...")
    
    if len(valid_neurons) == 0:
        print("  No valid neurons found. Trying to find any available neurons...")
        # Find first available neuron
        for layer_idx in sorted(neuron_stats.keys()):
            if len(neuron_stats[layer_idx]) > 0:
                first_neuron = min(neuron_stats[layer_idx].keys())
                valid_neurons.append((layer_idx, first_neuron))
                print(f"  Found: Layer {layer_idx}, Neuron {first_neuron}")
                break
    
    # Analyze each selected neuron
    for layer, neuron in valid_neurons:
        print(f"\n  Analyzing Layer {layer}, Neuron {neuron}...")
        
        # Print top-k tokens
        try:
            top_tokens = top_k_tokens_for_neuron(
                layer=layer,
                neuron=neuron,
                activations=loaded_activations,
                tokenizer=tokenizer,
                k=20
            )
            print(f"    Top 20 tokens for Layer {layer}, Neuron {neuron}:")
            for i, token_data in enumerate(top_tokens[:10], 1):
                if len(token_data) == 6:
                    token_str, activation_val, token_id, batch_idx, batch_item, seq_pos = token_data
                    print(f"      {i:2d}. '{token_str}' (activation: {activation_val:.4f}, token_id: {token_id}, batch={batch_idx}, seq={batch_item}, pos={seq_pos})")
                else:
                    # Backwards compatibility with old format
                    token_str, activation_val, token_id, batch_idx, seq_pos = token_data[:5]
                    print(f"      {i:2d}. '{token_str}' (activation: {activation_val:.4f}, token_id: {token_id})")
            
            # Plot top-k tokens
            try:
                topk_path = f"topk_tokens_l{layer}_n{neuron}.png"
                plot_topk_tokens(layer, neuron, top_tokens, topk_path)
                print(f"    Top-k tokens plot saved to {topk_path}")
            except Exception as e:
                print(f"    Error creating top-k tokens plot: {e}")
        except Exception as e:
            print(f"    Error getting top tokens: {e}")
        
        # Save histogram plot
        try:
            fig = visualize_neuron_distribution(layer, neuron, loaded_activations, 
                                                 by_position=args.positional_histograms)
            hist_path = f"neuron_l{layer}_n{neuron}_histogram.png"
            fig.savefig(hist_path)
            print(f"    Histogram saved to {hist_path}")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"    Error creating histogram: {e}")
        
        # Save token activation heatmap
        try:
            fig = visualize_token_activation_heatmap(layer, neuron, loaded_activations)
            heatmap_path = f"token_activation_heatmap_l{layer}_n{neuron}.png"
            fig.savefig(heatmap_path)
            print(f"    Token activation heatmap saved to {heatmap_path}")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"    Error creating token activation heatmap: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Interpretability pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

