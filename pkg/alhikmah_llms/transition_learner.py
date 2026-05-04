"""
A tiny learner that discovers transition probabilities from data via gradient descent.

Students don't need to read this file yet. The internals (softmax, cross-entropy,
gradients) will be explained in the bigram notebook. For now, treat it as a black box:
start from random guesses, call train(), get probabilities.
"""

import torch
import torch.nn.functional as F


class TransitionLearner:
    """Learns an NxN transition probability matrix from sequences of integers."""

    def __init__(self, n, seed=42):
        torch.manual_seed(seed)
        self.n = n
        self.W = torch.zeros((n, n), requires_grad=True)

    def probabilities(self):
        """Return the current NxN probability matrix as a list of lists."""
        with torch.no_grad():
            return F.softmax(self.W, dim=1).tolist()

    def train(self, xs, ys, steps=200, lr=10.0, print_every=None):
        """Train on (input, target) pairs. Optionally print loss at given interval."""
        xs_t = torch.tensor(xs)
        ys_t = torch.tensor(ys)

        for step in range(steps):
            logits = self.W[xs_t]
            loss = F.cross_entropy(logits, ys_t)

            if print_every is not None and step % print_every == 0:
                print(f'  step {step:>3d}  loss = {loss.item():.4f}')

            self.W.grad = None
            loss.backward()
            with torch.no_grad():
                self.W -= lr * self.W.grad
