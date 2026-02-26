#!/usr/bin/python3
"""
train2.py: Bigram language model with a single-layer MLP, trained with autograd.

Same as train1.py:
- Dataset, tokenizer, model architecture (MLP), SGD optimizer, inference

Different from train1.py:
- Introduces autograd (Value class) to compute gradients automatically
- No more manual analytic_gradient or numerical_gradient
- The forward pass builds a computation graph, loss.backward() traverses it

The hand-derived chain rule from train1.py is now automated: each operation
records its local gradient, and backward() applies the chain rule recursively.
The training loop becomes: forward pass -> loss.backward() -> SGD update.
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
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

# Autograd: recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters
n_embd = 16     # embedding dimension
matrix = lambda nout, nin: [[Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'mlp_fc1': matrix(4 * n_embd, n_embd),
    'mlp_fc2': matrix(vocab_size, 4 * n_embd),
}
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# Model: token_id -> logits
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    probs = [e / total for e in exps]
    return probs

def mlp(token_id):
    x = state_dict['wte'][token_id]
    x = linear(x, state_dict['mlp_fc1'])
    x = [xi.relu() for xi in x]
    logits = linear(x, state_dict['mlp_fc2'])
    return logits

# Train the model
num_steps = 1000
learning_rate = 1.0
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = len(tokens) - 1

    # Forward pass
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = mlp(token_id)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    # Backward pass
    loss.backward()

    # SGD update
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        p.data -= lr_t * p.grad
        p.grad = 0

    # if step < 5 or step % 200 == 0:
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# Inference: sample new names from the model
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    token_id = BOS
    sample = []
    for pos_id in range(16):
        logits = mlp(token_id)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
