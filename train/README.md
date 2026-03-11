# Intermediate implementations

This folder contains some of the preliminary implementations as [listed here](https://karpathy.github.io/2026/02/12/microgpt/#progression):

- train0: Bigram count table — no neural net, no gradients
- train1: MLP + manual gradients (numerical & analytic) + SGD
- train2: Autograd (Value class) — replaces manual gradients
- train3: Position embeddings + single-head attention + rmsnorm + residuals
- train4: Multi-head attention + layer loop — full GPT architecture
- train5: Adam optimizer
