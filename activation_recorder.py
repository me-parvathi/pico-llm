"""
Utilities for collecting, saving, and loading model activations.
"""

import torch


def collect_activations(model, tokenizer, dataset, max_batches=None):
    """
    Collect activations from the model over the dataset.
    
    Args:
        model: The trained pico-llm model (should support return_activations=True)
        tokenizer: Tokenizer/encoder object (e.g., tiktoken encoder)
        dataset: PyTorch DataLoader or iterable that yields batches of tokens
                 Each batch should be shape (seq_len, batch_size)
        max_batches: Maximum number of batches to process (None = process all)
    
    Returns:
        dict with keys:
            - "token_ids": list of token id tensors
            - "raw_text": list of raw text strings
            - "activations": list of activation dicts (one per batch)
    """
    model.eval()
    
    # Storage for collected data
    all_token_ids = []
    all_raw_text = []
    all_activations = []
    
    batch_count = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, batch_tokens in enumerate(dataset):
            if max_batches is not None and batch_count >= max_batches:
                break
            
            # Move batch to device if needed
            batch_tokens = batch_tokens.to(device)
            seq_len, batch_size = batch_tokens.shape
            
            # Decode raw text for this batch
            batch_texts = []
            batch_token_ids_list = []
            for b in range(batch_size):
                # Extract sequence for this batch item (remove padding)
                seq_tokens = batch_tokens[:, b]
                # Find actual length (first padding token = 0)
                non_padding = (seq_tokens != 0).nonzero(as_tuple=True)[0]
                if len(non_padding) > 0:
                    actual_len = non_padding[-1].item() + 1
                    seq_tokens = seq_tokens[:actual_len]
                else:
                    seq_tokens = seq_tokens[seq_tokens != 0]
                
                if len(seq_tokens) > 0:
                    token_ids_list = seq_tokens.cpu().tolist()
                    text = tokenizer.decode(token_ids_list)
                    batch_texts.append(text)
                    batch_token_ids_list.append(seq_tokens.cpu())
                else:
                    batch_texts.append("")
                    batch_token_ids_list.append(torch.tensor([], dtype=torch.long))
            
            # Run model with return_activations=True
            logits, activations = model(batch_tokens, return_activations=True)
            
            # Store data for this batch
            all_token_ids.extend(batch_token_ids_list)
            all_raw_text.extend(batch_texts)
            all_activations.append(activations)
            
            batch_count += 1
    
    return {
        "token_ids": all_token_ids,
        "raw_text": all_raw_text,
        "activations": all_activations
    }


def save_activations(path, data_dict):
    """
    Save activations to disk as a compressed .pt file.
    
    Args:
        path: File path to save to (should end with .pt)
        data_dict: Dictionary containing activations data (from collect_activations)
    """
    torch.save(data_dict, path, _use_new_zipfile_serialization=True)


def load_activations(path):
    """
    Load activations from disk.
    
    Args:
        path: File path to load from (.pt file)
    
    Returns:
        Dictionary containing activations data
    """
    return torch.load(path, map_location='cpu')

