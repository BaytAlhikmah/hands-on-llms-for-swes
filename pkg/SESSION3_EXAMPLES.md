# Session 3 Visualization Examples

This document shows how to use the Session 3 visualization functions for teaching cross-entropy and KL divergence.

## Installation

Make sure the package is installed:
```bash
cd pkg/
pip install -e .
```

## Basic Usage

```python
from alhikmah_llms import Session3
import matplotlib.pyplot as plt
```

## 1. Core Intuition: Comparing Distributions

Show the fundamental concept - cross-entropy measures how well predicted distribution matches true distribution:

```python
# Default example: predicting next character after 'q'
Session3.plot_cross_entropy_intuition()
plt.show()

# Custom example
Session3.plot_cross_entropy_intuition(
    true_probs=[0.6, 0.3, 0.1],
    predicted_bad=[0.33, 0.33, 0.34],
    predicted_good=[0.58, 0.32, 0.1],
    labels=['Rock', 'Paper', 'Scissors']
)
plt.show()
```

## 2. Surprise Breakdown

Show which outcomes contribute most to the loss:

```python
Session3.plot_surprise_breakdown()
plt.show()

# Custom
Session3.plot_surprise_breakdown(
    true_probs=[0.7, 0.2, 0.07, 0.03],
    predicted=[0.5, 0.3, 0.15, 0.05],
    labels=['u', 'a', 'e', 'i']
)
plt.show()
```

## 3. Entropy Decomposition

The key insight: H(P,Q) = H(P) + KL(P||Q)

```python
Session3.plot_entropy_decomposition()
plt.show()

# Custom
Session3.plot_entropy_decomposition(
    true_probs=[0.8, 0.15, 0.05],
    predicted=[0.6, 0.25, 0.15]
)
plt.show()
```

## 4. Training Trajectory

Watch the model converge during training:

```python
Session3.plot_training_trajectory()
plt.show()

# Custom with your own training steps
steps = [
    [0.33, 0.33, 0.34],  # initialization
    [0.50, 0.30, 0.20],  # epoch 10
    [0.65, 0.25, 0.10],  # epoch 50
    [0.75, 0.20, 0.05],  # epoch 100
]
Session3.plot_training_trajectory(
    true_probs=[0.8, 0.15, 0.05],
    steps=steps
)
plt.show()
```

## 5. KL Divergence Asymmetry

Show that KL(P||Q) ≠ KL(Q||P):

```python
Session3.plot_kl_asymmetry()
plt.show()

# Custom
Session3.plot_kl_asymmetry(
    P=[0.9, 0.08, 0.02],  # very peaked
    Q=[0.4, 0.4, 0.2],     # flatter
    labels=['A', 'B', 'C']
)
plt.show()
```

## 6. Cross-Entropy Loss Surface

Visualize the loss landscape:

```python
Session3.plot_cross_entropy_surface()
plt.show()

# Custom true distribution
Session3.plot_cross_entropy_surface(
    true_probs=[0.5, 0.3, 0.2]
)
plt.show()
```

## 7. Bigram Example (Real Use Case)

Show cross-entropy during actual model training:

```python
# After 'q', what comes next?
Session3.plot_bigram_cross_entropy_example()
plt.show()

# Custom example
Session3.plot_bigram_cross_entropy_example(
    true_counts=[100, 50, 20, 10, 5],
    labels=['e', 'a', 'i', 'o', 'u'],
    predictions={
        'Untrained': [0.2, 0.2, 0.2, 0.2, 0.2],
        'Early training': [0.4, 0.25, 0.15, 0.12, 0.08],
        'Converged': [0.54, 0.27, 0.11, 0.05, 0.03],
    }
)
plt.show()
```

## 8. Loss Curve

Visualize training progress:

```python
# Generate example loss curve
Session3.plot_loss_curve()
plt.show()

# With real training data
import numpy as np

train_losses = [2.5, 2.1, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02, 1.0]
val_losses = [2.6, 2.2, 1.9, 1.6, 1.4, 1.3, 1.25, 1.2, 1.18, 1.15]

Session3.plot_loss_curve(
    losses=train_losses,
    val_losses=val_losses
)
plt.show()
```

## Recommended Order for Teaching

1. **Start with intuition**: `plot_cross_entropy_intuition()` - show the concept visually
2. **Break it down**: `plot_surprise_breakdown()` - where does the loss come from?
3. **The key insight**: `plot_entropy_decomposition()` - you can't reduce H(P), only minimize KL!
4. **Watch it learn**: `plot_training_trajectory()` - Q moves toward P
5. **Real example**: `plot_bigram_cross_entropy_example()` - concrete bigram model
6. **Advanced**: `plot_kl_asymmetry()` and `plot_cross_entropy_surface()` - deeper understanding

## In a Notebook

```python
from alhikmah_llms import Session3
import matplotlib.pyplot as plt

# Exercise 1: Understanding cross-entropy
print("Exercise 1: What is cross-entropy?")
Session3.plot_cross_entropy_intuition()
plt.show()

# Exercise 2: Where does the loss come from?
print("Exercise 2: Loss breakdown")
Session3.plot_surprise_breakdown()
plt.show()

# Exercise 3: The decomposition
print("Exercise 3: H(P,Q) = H(P) + KL(P||Q)")
Session3.plot_entropy_decomposition()
plt.show()

# And so on...
```

## Notes

- All functions use **natural log (ln)** by default, so loss is measured in **nats**, not bits
- This matches PyTorch's `nn.CrossEntropyLoss()` and standard ML practice
- To convert: `bits = nats / ln(2) ≈ nats * 1.443`
- Colors and styling match Session 2's pedagogical approach
- All visualizations work in Jupyter, Colab, and standard Python scripts
