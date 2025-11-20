# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

# Some Command Line Arguments you can use
# python -m venv .venv
# python pico_llm.py --block_size 32 --tinystories_weight 0.0 --input_files 3seqs.txt --prompt "0 1 2 3 4" --device_id cuda:0
# python pico_llm.py --block_size 32 --tinystories_weight 1.0 --prompt "Once upon a time" --device_id cuda:0

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    parser.add_argument("--plot_nucleus_debug", action="store_true",
                        help="If set, plot nucleus sampling distribution during generation (for debugging).")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # fill in
        #############################################
        #############################################

        # calculate the Input Dimensions and Vector Size. In this kGram part, we're NOT using nn.Embedding.
        # we're instead doing this the dumb way, where we use a Sparse Matrix of size k* vocabsize (which is a massive overhead)
        # Also add Hidden Dimensions, and Number of Inner Layers
        input_dim = self.k * self.vocab_size
        hidden_dim = self.embed_size
        output_dim = self.vocab_size

        # Start making the layers, Make Linear Layers the size and depth of the input/hidden_dim
        # below is the "input layer"
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU()) # Or nn.SiLU()

        # Add the specified number of hidden inner layers
        # This makes the Neural net "Deep". However the default is literally "1"!
        for _ in range(num_inner_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU()) # Or nn.SiLU()

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # The forward() loop will call this!
        self.net = nn.Sequential(*layers)
        
        #############################################
        #############################################

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs


'''
    Notes: Upon some research, I realized that this above model is an ancient "failed" attempt at training a next token predictor.
    Basically, since we flatten the vector with vocabulary size of 50k, and then also add a hidden layer of size 50k*1024
    (or whatever thw batch size is) we see that for each token, the GPU (I have it enabled) performs a matrix multiplication of size 150Million!
    That is not only computationally expensive, it is practically infeasible for most applications.
    In fact LSTMs performed better because it's forward method didn't have hard coded loops which uses CPU
    Bottom Line: We understand that the assignment is trying to help us learn that K-Gram is computationally infeasible!
    Note: We will run this and LSTMs in front of graders and show them the difference in performance.
'''

################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization (RMSNorm).
    
    A simpler and faster alternative to LayerNorm that only
    scales the input, without centering it (no mean subtraction).
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        
        # gain, initialized to all "1"s
        self.gain = nn.Parameter(torch.ones(dim))
        
        # Epsolpn to prevent div by zero
        self.eps = eps

    def forward(self, x):
        # Calculate the RMS of the input - We compute 1.0 / sqrt(mean(x^2) + eps)
        rms_dev = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize and apply gain
        return x * rms_dev * self.gain
    
# Note: RMSNorm is the simpler alternative to LayerNorm or BatchNorm.
# It just has a single "gain" factor and doesn't try to zero-center the mean!

################################################################
####            Transformer Architecture                    ####
################################################################

# First we write an "Attention" Mechanism, and call it the Multi Head Attention, where we will combine many single Self Attention Layers 
class MultiHeadAttention(nn.Module):
    """
    Implementation of Causal Multi-Head Self-Attention (part of Task 4c).
    
    This is the "communication" layer. It lets tokens "look at"
    and gather information from previous tokens.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Linear layers to project input into Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, return_attn=False):
        # Input
        seq_len, batch_size, d_model = x.shape
        
        # Project to Q, K, V, and Split it
        qkv = self.qkv_proj(x) # (seq_len, batch, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1) # Each is (seq_len, batch, d_model)
        
        # Reshape and transpose for multi-head
        q = q.view(seq_len, batch_size, self.n_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        k = k.view(seq_len, batch_size, self.n_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        v = v.view(seq_len, batch_size, self.n_heads, self.head_dim).transpose(0, 1).transpose(1, 2)

        # (B, nH, T, H) @ (B, nH, H, T) -> (B, nH, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # This code prevents the model from "cheating" by looking at future tokens, also called causal Look-Ahead Mask
        T = seq_len
        # torch.triu creates an upper-triangular matrix
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -float('inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # (B, nH, T, T) @ (B, nH, T, H) -> (B, nH, T, H)
        context = attn_weights @ v
        
        # (B, nH, T, H) -> (B, T, nH, H) -> (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        context = context.transpose(0, 1)

        output = self.out_proj(context)
        
        if return_attn:
            return output, attn_weights
        else:
            return output

# Following is a single Transformer Head, which we will combine as a "block" in the main model
class TransformerBlock(nn.Module):
    """
    A single Transformer Block.
    
    This is the "thinking" unit, combining communication (attention)
    and computation (MLP) with normalization and residual connections.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        # First sub-layer: Causal Multi-Head Attention
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        
        # Second sub-layer: Feed-Forward Network (MLP)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), # Standard 4x expansion
            nn.SiLU(), # Choice: Use ReLU vs Swish like a choice variable
            nn.Linear(d_model * 4, d_model)
        )
        self.final_norm = RMSNorm(d_model)

    def forward(self, x, return_attn=False, return_activations=False):
        # x shape: (seq_len, batch, d_model)
        # We have used a Pre-Normalization architecture (like GPT-2 and Llama)
        
        # Attention Sub-layer, which also has the residual connection
        # (x = x + Sum(f(x_j)))
        # x = x + self.attn(self.norm1(x))
        # z = z + g(z)
        # x = x + self.mlp(self.norm2(x))
        attn_output = self.attn(x, return_attn=(return_attn or return_activations))
        if return_attn or return_activations:
            attn_out, attn_weights = attn_output
            x = x + attn_out
        else:
            x = x + attn_output
        
        # Collect pre-MLP activation if needed
        mlp_pre = None
        if return_activations:
            # Pre-MLP activation: after first Linear layer, before SiLU
            # Note: matching actual code execution where self.mlp(x) is called directly
            mlp_pre = self.mlp[0](x)  # (seq_len, batch, d_model * 4)
        
        x = x + self.mlp(x)   
        
        x = self.final_norm(x)
        
        if return_activations:
            return x, attn_weights, mlp_pre
        elif return_attn:
            return x, attn_weights
        else:
            return x

# Finally, we assemble multiple Transformer blocks
class TransformerModel(nn.Module):
    """
    A full Causal Decoder-Only Transformer Achitecture.
    """
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4, block_size=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # Token Embedding Layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Embedding Layer
        # We need a vector for each position from 0 to block_size-1
        self.pos_embedding = nn.Embedding(block_size, d_model)
        
        # A stack of Transformer Blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_blocks)]
        )
        
        # Final normalization (Llama-style)
        self.final_norm = RMSNorm(d_model)
        
        # Final "unembedding" layer to get logits
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens_seq, return_attn=False, return_activations=False):
        # tokens_seq: (seq_len, batch)
        seq_len, batch_size = tokens_seq.shape
        
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds model's block size {self.block_size}")
        
        # Get token embeddings
        tok_emb = self.token_embedding(tokens_seq) # (seq_len, batch, d_model)
        
        # Get positional embeddings
        # Create position IDs: [0, 1, 2, ..., seq_len-1]
        # Then add them all together
        pos = torch.arange(0, seq_len, dtype=torch.long, device=tokens_seq.device).unsqueeze(1) # (seq_len, 1)
        pos_emb = self.pos_embedding(pos) # (seq_len, 1, d_model)

        x = tok_emb + pos_emb
        
        # Initialize activations dict if needed
        activations = None
        if return_activations:
            activations = {
                "embeddings": tok_emb,
                "layers": []
            }
        
        # Run through all Transformer blocks
        for block in self.blocks:
            if return_activations:
                block_output = block(x, return_attn=False, return_activations=True)
                x, attn_weights, mlp_pre = block_output
                activations["layers"].append({
                    "attention": attn_weights,
                    "mlp_pre": mlp_pre
                })
            elif return_attn:
                block_output = block(x, return_attn=True, return_activations=False)
                x, attn_weights = block_output
            else:
                x = block(x, return_attn=False, return_activations=False)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        # Apply unembedding layer to get logits
        logits = self.lm_head(x) # (seq_len, batch, vocab_size)
        
        if return_activations:
            return logits, activations
        elif return_attn:
            return logits, attn_weights
        else:
            return logits


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95, plot_debug=False):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending = True)
    
    if plot_debug:
        from plots import plot_nucleus_distribution
        plot_nucleus_distribution(sorted_probs, p, save_path="nucleus.png")
    
    cumulative_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumulative_sum_probs <= p 

    if not nucleus_mask.any():
        nucleus_mask[0] = True
    else:
        last_idx = torch.where(nucleus_mask)[0][-1].item()
        if last_idx < len(nucleus_mask) - 1:
            nucleus_mask[last_idx + 1] = True
    
    nucleus_probs = sorted_probs[nucleus_mask]
    nucleus_indices = sorted_indices[nucleus_mask]

    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    sampled_idx = torch.multinomial(nucleus_probs, num_samples=1)
    chosen_token = nucleus_indices[sampled_idx].item()

    return chosen_token


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False,
                  plot_nucleus_debug=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p, plot_debug=plot_nucleus_debug)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


################################################################################
# 8. Checkpointing
################################################################################

def save_checkpoint(model, optimizer, path):
    checkpoint = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict()}
    # Save model config if it's a TransformerModel
    if hasattr(model, 'vocab_size') and hasattr(model, 'd_model'):
        checkpoint["config"] = {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "n_blocks": model.n_blocks,
            "block_size": model.block_size
        }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return model, optimizer


################################################################################
# 9. Training
################################################################################
from torch.optim.lr_scheduler import OneCycleLR

# This function uses the 
def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    batch_losses=None,
                    epoch_losses=None,
                    plot_nucleus_debug=False,
                    checkpoint_dir="checkpoints",
                    checkpoint_steps=50):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    # AdamW is 'Adam with Weight Decay fixed'.
    # weight_decay=0.1 or 0.01 are standard values.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0
    
    # Calculate steps per epoch correctly
    if max_steps_per_epoch is not None:
        steps_per_epoch = min(len(loader), max_steps_per_epoch)
    else:
        steps_per_epoch = len(loader)

    total_steps = steps_per_epoch * epochs

    # This is for the "Warmup" where the learning rate starts from 0, goes to the max which we set (3e-4) and then goes back down
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            if batch_losses is not None:
                batch_losses[model_name].append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            # Clips gradients to have a max norm of 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step the Optimizer and the Scheduler 
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None),
                        plot_nucleus_debug=plot_nucleus_debug
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None),
                        plot_nucleus_debug=plot_nucleus_debug
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None),
                        plot_nucleus_debug=plot_nucleus_debug
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

            # Save checkpoint every N steps if specified
            if checkpoint_dir is not None and checkpoint_steps is not None and global_step % checkpoint_steps == 0:
                import os
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_step_{global_step}.pt")
                save_checkpoint(model, optimizer, checkpoint_path)
                print(f"[{model_name}] Saved checkpoint at step {global_step} to {checkpoint_path}")

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")
        
        if epoch_losses is not None:
            epoch_losses[model_name].append(avg_loss)

        # Save checkpoint after each epoch
        if checkpoint_dir is not None:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, checkpoint_path)
            print(f"[{model_name}] Saved checkpoint after epoch {epoch} to {checkpoint_path}")

# This is used to overfit the model. It basically use uses 100 steps and works on a very small/single batch
def train_overfit(model,
                  loader,
                  steps=100,
                  model_name="overfit",
                  batch_losses=None,
                  device="cpu",
                  lr=3e-4):
        """
        Overfit function: Grabs a SINGLE batch from the loader and trains on it repeatedly.
        If the model architecture is correct, loss should drop to near 0.0.
        """
        print(f"\n[{model_name}] Starting OVERFIT test on device: {device}")
        
        # 1. Setup Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()

        # 2. Grab ONE single batch
        try:
            first_batch = next(iter(loader))
        except StopIteration:
            print(f"[{model_name}] Error: Loader is empty!")
            return

        # Move to device ONCE
        # batch_tokens shape: (seq_len, batch_size)
        batch_tokens = first_batch.to(device)
        
        print(f"[{model_name}] Overfitting on a single batch of shape: {batch_tokens.shape}")
        
        start_time = time.time()

        # 3. The Overfit Loop
        for step in range(1, steps + 1):
            # A. Forward Pass
            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            
            # B. Compute Loss
            # Note: compute_next_token_loss handles the shifting internally
            loss = compute_next_token_loss(logits, batch_tokens)

            # C. Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # D. Logging (Every 10 steps)
            if step % 10 == 0 or step == 1:
                print(f"[{model_name}] Step {step}/{steps} | Loss: {loss.item():.6f}")
                if batch_losses is not None:
                    batch_losses[model_name].append(loss.item())

        total_time = time.time() - start_time
        print(f"[{model_name}] Overfit test complete. Final Loss: {loss.item():.6f} (Time: {total_time:.2f}s)")
        
################################################################################
# 10. Main
################################################################################

def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 16
    num_epochs = 3
    learning_rate = 3e-4

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
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

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    # This part uses the collation function, and merges datasets as you want
    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    # K-Gram, won't really be used, but added to showcase how One Hot Encofding is a computational disaster
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    # LSTM, serves as a baseline Sanity Check
    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    # A baseline Transformer Architecture, serves as the benchmark, if you get it to converge lol 
    # In your main script or config
    transformer = TransformerModel(
        vocab_size=50257,
        d_model=768,
        n_heads=12,
        n_blocks=6,
        block_size=64
    ).to(device)

    # The list of normal Models to overfit
    overfit_models = {
        #"kgram_mlp_seq_overfit": kgram_model,
        "lstm_seq_overfit": lstm_model,
        "transformer_overfit": transformer
    }
    
    # The list of normal Models to train and converge
    models = {
        #"kgram_mlp_seq": kgram_model,
        "lstm_seq": lstm_model,
        "transformer": transformer
    }

    # The Dictionaries that you can use to Plot the Losses in plots.py 
    batch_losses = {model_name: [] for model_name in models}
    overfit_batch_losses = {model_name: [] for model_name in overfit_models}
    epoch_losses = {model_name: [] for model_name in models}

    ############################################################################
    # Train each model
    ############################################################################

    # Train all the Models and check whether they appropriately overfit
    for model_name, model in overfit_models.items():
        print(f"\n=== Overfitting model: {model_name} ===")
        train_overfit(
            model=model, 
            loader=train_loader, 
            steps=100, 
            model_name=model_name,
            batch_losses=overfit_batch_losses,
            device=device
        )

    # Train all the Models and check whether they converge appropriately
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,  # <--- Pass the user-specified prompt here
            batch_losses=batch_losses,
            epoch_losses=epoch_losses,
            plot_nucleus_debug=args.plot_nucleus_debug
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
                plot_nucleus_debug=args.plot_nucleus_debug
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
                plot_nucleus_debug=args.plot_nucleus_debug
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
                plot_nucleus_debug=args.plot_nucleus_debug
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")
        
        # Plot attention heatmap for transformer model (debugging)
        if model_name == "transformer":
            with torch.no_grad():
                seq_tensor = torch.tensor(enc.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(1)
                logits, attn = model(seq_tensor, return_attn=True)
                # Extract head 0, batch 0: attn shape is (B, nH, T, T)
                attn_head0 = attn[0, 0].cpu().numpy()  # (seq_len, seq_len)
                
                # Debug: print attention statistics
                print(f"[{model_name}] Attention stats - min: {attn_head0.min():.6f}, max: {attn_head0.max():.6f}, mean: {attn_head0.mean():.6f}")
                print(f"[{model_name}] Attention shape: {attn_head0.shape}")
                
                from plots import plot_attention_heatmap
                plot_attention_heatmap(attn_head0, save_path="attn.png")
                print(f"[{model_name}] Attention heatmap saved to attn.png")
    

    # Plot all the Models, including the Overfit ones. The Epoch Loss tells us a bigger picture
    from plots import plot_batch_losses, plot_epoch_losses
    plot_batch_losses(overfit_batch_losses, save_path="overfit_batch_loss.png")
    plot_batch_losses(batch_losses, save_path="batch_loss.png")
    plot_epoch_losses(epoch_losses, save_path="epoch_loss.png")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()
