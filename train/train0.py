#!/usr/bin/python3
"""
train0.py: Bigram language model trained by counting.

The structure is identical to train.py: tokenize, forward pass, update, sample.
The only difference is what's inside the "model" box:
- train.py: gpt(token_id) -> logits, trained by gradient descent
- train0.py: bigram(token_id) -> probs, trained by counting

A bigram model is a special case of a GPT where there is no attention (each token
only looks at itself), no MLP, and the "embedding" is just a row in a lookup table.
Counting is the closed-form solution for this case; gradient descent is what you
need when the model is too expressive for exact solutions.
"""

import os       # os.path.exists
import math     # math.log
import random   # random.seed, random.choices, random.shuffle
random.seed(42)

# Dataset: load and tokenize a list of names
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()] # list[str] of documents
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer: character-level, with a special BOS (Beginning of Sequence) token
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Initialize the parameters: a bigram count table. state_dict[i][j] = how many times token j follows token i
state_dict = [[0] * vocab_size for _ in range(vocab_size)]

# The "model": given a token_id, return the probability distribution over the next token
def bigram(token_id):
    row = state_dict[token_id]
    total = sum(row) + vocab_size # add-one (Laplace) smoothing
    return [(c + 1) / total for c in row]

# Train the model
num_steps = 1000
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = len(tokens) - 1

    # Forward pass: compute the loss for this document
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        probs = bigram(token_id)
        loss_t = -math.log(probs[target_id])
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    # Update the model: incorporate this document's bigram counts
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        state_dict[token_id][target_id] += 1

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss:.4f}")

# Inference: sample new names from the model
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    token_id = BOS
    sample = []
    for _ in range(16): # maximum sequence length
        token_id = random.choices(range(vocab_size), weights=bigram(token_id))[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
