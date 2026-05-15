In session 1, you saw that a language model does one thing: predict the next token. But how do you measure whether those predictions are any good? And how predictable is language in the first place?

In 1948, Claude Shannon answered both questions with a single idea. This session builds that idea from scratch — starting with entropy (the measure of unpredictability), then cross-entropy (the measure of prediction quality), and finally KL divergence (what training actually optimizes).

Read this alongside `notebook.ipynb` — every hands-on section below maps to a cell you should run.

## Learning Objectives

By the end of this session, you will understand:

- What **entropy** is, in three different ways: average yes/no questions to guess an outcome, optimal compression size, and average surprise
- How to compute entropy from a probability distribution, and what "bits" actually measure
- What **cross-entropy** is and why it's the loss function behind every language model
- The decomposition **H(P,Q) = H(P) + KL(P||Q)** and why training only minimizes the KL divergence term
- What **KL divergence** measures and why it's not symmetric
- The difference between **bits** (log₂) and **nats** (ln), and why ML uses nats
- What "loss" means: average surprise of the model on real data

But first, where did these ideas come from? And why do they matter?

---

## 0. How to Measure Information

It's 1948, and Claude Shannon has just published what will become the most important paper in the history of digital communication. Working at Bell Labs, he's been obsessed with a single question: **How much information is in a message?**

The telephone company needs to know: How many wires do we need? How much can we compress a signal before it becomes gibberish? When static corrupts a message, how much was actually lost?

But nobody knew how to even **measure** information. Weight? Length? Number of words? None of these captured the essence of what information really is.

Shannon's insight: **Information is the reduction of uncertainty.**

If I tell you something you already knew, that's zero information. If I tell you something completely surprising, that's maximum information. The more uncertain you were before, the more information you gain when you learn the answer.

He formalized this with a mathematical measure called **entropy** — a single number that captures how unpredictable something is. The formula became the foundation of information theory, compression algorithms, cryptography, and modern machine learning.

To understand what entropy means and how it works, let's start with a simple example.

---

## 1. Entropy: The Guessing Game

Before we touch any code, let's build an intuition.

**Setup:** I'm thinking of a number between 1 and 8. You can ask me yes/no questions. How many questions do you need?

If you play optimally (binary search):
- "Is it > 4?" → "No"
- "Is it > 2?" → "Yes"
- "Is it = 3?" → "No"

**Answer: 4.** Took **3 questions**.

This isn't luck. With 8 equally likely options, an optimal strategy needs exactly **log₂(8) = 3** yes/no questions on average — which matches the **entropy** of the distribution.

> **Notebook Exercise 1.** Visualize the binary decision tree for the uniform case (1-8).

Now try a different game. Same setup, but this time I tell you: "50% of the time the number is 1. The other 50% is split equally among 2–8."

Your strategy changes:
- "Is it 1?" → 50% of the time: **done in 1 question**. 50% of the time: "No" → now you have 7 equally likely options left, which takes log₂(7) ≈ 2.807 more questions.

**Average questions needed:**
```
0.5 × 1  +  0.5 × (1 + 2.807)  =  2.4035 questions
```

> **Notebook Exercise 2.** Visualize the binary decision tree for the biased case and compare with the uniform case.

> **Notebook Exercise 3.** Visualize the probability distributions and see how entropy relates to the "spread" of probabilities.

---

### Why "2.4035 questions"? Don't questions give yes/no answers?

Yes! But we're computing the **expected value** (average across many games). Sometimes you get lucky (1 question), sometimes you need 4. On average: 2.4035 questions.

---

### The Same Calculation, Three Ways

**Method 1: Grouped by outcome** (what we just did)
```
0.5 × 1  +  0.5 × (1 + 2.807)  =  2.4035
 ↑            ↑
 |            └─ "No" branch: 1 initial + 2.807 more = 3.807 total
 └─ "Yes" branch: done in 1
```

**Method 2: Each number individually**

Instead of grouping "2–8" together, let's count each one:
```
Number 1: probability 0.5,   needs 1 question       → 0.5 × 1 = 0.500
Number 2: probability 0.071, needs 3.807 questions  → 0.071 × 3.807 = 0.272
Number 3: probability 0.071, needs 3.807 questions  → 0.071 × 3.807 = 0.272
Number 4: probability 0.071, needs 3.807 questions  → 0.071 × 3.807 = 0.272
...
Number 8: probability 0.071, needs 3.807 questions  → 0.071 × 3.807 = 0.272
                                                        ─────────────────────
                                                        Total: 2.4035 questions
```

Or algebraically:
```
0.5 × 1  +  (0.5/7) × 3.807  +  (0.5/7) × 3.807  +  ...  +  (0.5/7) × 3.807
                                                              └─ 7 times

= 0.5 × 1  +  7 × (0.5/7) × 3.807
= 0.5 × 1  +  0.5 × 3.807
= 2.4035 questions
```

**Method 3: The entropy formula**

Each number has probability P and needs log₂(1/P) yes/no questions. Substituting:

- Number 1: P = 0.5, so log₂(1/0.5) = log₂(2) = 1
- Numbers 2–8: total P = 0.5 (split equally), so log₂(1 / (0.5/7)) = log₂(14) ≈ 3.807

```
0.5 × log₂(2)  +  7 × (0.5/7) × log₂(14)
= 0.5 × 1      +  0.5 × 3.807
= 2.4035 bits
```

**Why "1 + log₂(7) = log₂(14)"?** Because once you know the answer is "No" (not number 1), you need 1 question to confirm "No" + log₂(7) questions to identify which of the 7 it is. So log₂(1/0.5) + log₂(7) = 1 + log₂(7). Both forms equal 3.807.

This is the general pattern:
```
Entropy = ∑ P(x) × log₂(1/P(x))
        = 0.5 × log₂(1 / 0.5) + (0.5/7) × log₂(1 / (0.5/7)) +  ...  +  (0.5/7) × log₂(1 / (0.5/7))
                                                                                        └─ 7 times
        = 2.4035 bits
```

**This IS the entropy formula!** We just derived it from first principles.

> **Notebook Exercise 4.** Compute entropy for different probability distributions using the formula.

---

**Key insight:** Lower entropy = more predictable = fewer questions needed.
- Uniform (all equal): 3.00 bits
- Biased (50% on one): 2.4035 bits
- Certain (100% on one): 0.00 bits

---

## 2. Entropy: The Formula

Here's the general formula. Don't memorize it — just notice what it does:

**Why "bits"?** A **bit** means one binary decision — one yes/no distinction. Because each yes/no question has two possible outcomes, entropy measured with log₂ naturally counts information in bits.

```python
import math

def entropy(probs: list[float]) -> float:
    """Average number of yes/no questions (bits) to guess the outcome."""
    return sum(p * math.log2(1/p) for p in probs if p > 0)

    # it's also written as
    # -sum(p * math.log2(p) for p in probs if p > 0)
    # Question: Why? How are both equivalent?
```

> **Notebook Exercise 5.** Implement and test the entropy function on various distributions.

**Three examples:**

```python
# Uniform: maximum uncertainty
uniform_8 = [1/8] * 8
print(f"Uniform: {entropy(uniform_8):.2f} bits")  # 3.00

# Biased: less uncertainty
biased = [0.5, 0.5/7, 0.5/7, 0.5/7, 0.5/7, 0.5/7, 0.5/7, 0.5/7]
print(f"Biased: {entropy(biased):.2f} bits")  # ~2.4

# Certain: no uncertainty
certain = [1.0, 0, 0, 0, 0, 0, 0, 0]
print(f"Certain: {entropy(certain):.2f} bits")  # 0.00
```

**Intuition check:**
- Maximum entropy → uniform distribution → no clue what's next
- Low entropy → biased distribution → you can guess well
- Zero entropy → deterministic → you know the answer

Entropy measures unpredictability. Now that we know how to measure the uncertainty in data, how do we measure how well a *model* captures that uncertainty?

## 3. Three Ways to Think About Entropy

Software engineers come from different backgrounds, so here are three equivalent framings. Pick the one that clicks for you — they're all the same math.

### A. Guessing Game (you just saw this)

Entropy = lower bound on average number of yes/no questions to identify the outcome.

### B. Compression

Entropy = optimal file size in bits per symbol.

If you have a file `AAAAAABBCD` (10 characters):
- Naive encoding: each letter gets 2 bits (00=A, 01=B, 10=C, 11=D) → 20 bits
- Smart encoding: assign short codes to frequent letters
  - A (60%): `0` (1 bit)
  - B (20%): `10` (2 bits)
  - C (10%): `110` (3 bits)
  - D (10%): `111` (3 bits)

Encoded: `0 0 0 0 0 0 10 10 110 111` = **16 bits** → **1.6 bits/char**

Entropy of `[0.6, 0.2, 0.1, 0.1]` = **1.57 bits** — our encoding is near-optimal!

> **Notebook Exercise 6.** Visualize how entropy-based compression saves space.

**Key insight:** if a sequence has low entropy, you can compress it well. If it has high entropy, it's already random-looking and incompressible.

### C. Surprise

Entropy = average surprise when you see the outcome.

When something with probability `p` happens, your **surprise** is:
```
surprise = log2(1 / p) = -log₂(p)
```

| Event | Probability | Surprise (bits) |
|-------|------------|-----------------|
| Sun rises | 0.99 | log2(1 / 0.99) ~= 0.01449 |
| Coin flip | 0.50 | log2(1 / 0.50) = 1.00 |
| Win lottery | 0.000001 | log2(1 / 0.000001) ~= 19.9315 |

Entropy is just the **expected surprise** — the average over all possible outcomes weighted by their probabilities.

> **Notebook Exercise 7.** Visualize the surprise curve and see how surprise relates to probability.

**This framing is critical for what follows:** cross-entropy measures the average surprise of a *model* on real data. A well-trained model isn't surprised by what actually happens.

> **Notebook Exercise 8.** Explore entropy across different probability distributions using interactive visualizations.

---

### Properties of Entropy

Now that you've seen entropy in action, here are its key mathematical properties. These aren't just formalities — they capture fundamental facts about information and uncertainty.

**1. Non-negativity**
```
H(X) ≥ 0
```
Entropy is always non-negative. It equals zero only when one outcome has probability 1 (complete certainty).

**2. Maximum Entropy**
```
H(X) ≤ log₂(n)
```
For a discrete random variable with n possible outcomes, entropy is maximized when all outcomes are equally likely (uniform distribution). Maximum = log₂(n) bits.

**3. Continuity**

H is a continuous function of the probability distribution. Small changes in probabilities lead to small changes in entropy.

**4. Symmetry**

H is invariant under permutation of outcomes. The order doesn't matter, only the probabilities.

**5. Additivity (for independent events)**
```
H(X,Y) = H(X) + H(Y)  [if X and Y are independent]
```
The joint entropy of two independent random variables equals the sum of their individual entropies.

**6. Chain Rule**
```
H(X,Y) = H(X) + H(Y|X)
```
Joint entropy equals the entropy of X plus the conditional entropy of Y given X.

**7. Conditioning Reduces Entropy**
```
H(Y|X) ≤ H(Y)
```
Knowing X can only reduce (or leave unchanged) uncertainty about Y. Information never increases uncertainty. Equality holds when X and Y are independent.

> **Notebook Exercise 8c.** See these properties demonstrated with concrete examples.

---

## 4. Cross-Entropy: Measuring Prediction Quality

Now we move from measuring the uncertainty *in the data* to measuring *how well a model predicts the data*.

Entropy H(P) tells you the average surprise when outcomes come from distribution P and you know P. But in machine learning, the true distribution is P while your model predicts a *different* distribution Q. How surprised is your model?

This is **cross-entropy**:

```
H(P,Q) = -∑ P(x) log Q(x)
```

It's the average surprise of model Q when reality follows P.

**Key examples:**
- If Q = P (perfect model): H(P,Q) = H(P) — minimum possible loss
- If Q is uniform (knows nothing): H(P,Q) = log(n) — high loss
- If Q is wrong (predicts the opposite): H(P,Q) is very high — maximum surprise

**Lower cross-entropy = better model.**

> **Notebook Exercise 9.** Visualize true vs predicted distributions and see how cross-entropy measures the gap.

> **Notebook Exercise 10.** See which outcomes contribute most to the cross-entropy loss.

> **Notebook Exercise 11.** Implement cross-entropy from scratch.

---

## 5. What Is "Loss"?

When a training loop prints "loss = 1.234", what does that number mean?

It's the **cross-entropy** — the average surprise of the model on the training data. Specifically, for each example:

1. The model outputs a predicted distribution Q over possible outcomes
2. Reality reveals the actual outcome (drawn from the true distribution P)
3. The surprise is `-log Q(actual outcome)`

Average this over all examples and you get the loss.

**Example:**
- Your model says: "After 'q', 'u' has probability 0.6"
- Reality: 'u' appears
- Surprise: -ln(0.6) = 0.511 nats

Do this for every prediction and average. That's the loss.

**Lower loss = less surprised = better model.**

If the model were perfect (assigns probability 1.0 to what happens, 0 to everything else), loss would be 0. If the model is terrible (assigns probability 0.01 to what happens), loss is high.

This is what the learning algorithm minimizes. Every time it adjusts parameters, it's trying to be less surprised by the data.

---

## 6. Bits vs Nats

In Part 0, we measured entropy in **bits** using log₂. In machine learning, we switch to **nats** using natural log (ln).

Why?
- PyTorch's `nn.CrossEntropyLoss()` uses natural log
- Derivatives are cleaner: d/dx(ln x) = 1/x (vs d/dx(log₂ x) = 1/(x ln 2))
- Standard in ML papers and frameworks

**Conversion:** bits = nats / ln(2) ≈ nats × 1.443

From now on, when we write cross-entropy and loss, we use **nats** (natural log) unless stated otherwise.

> **Notebook Exercise 12.** Compare cross-entropy in bits vs nats.

---

## 7. The Key Decomposition: H(P,Q) = H(P) + KL(P||Q)

Here's the most important equation in this session:

```
Cross-Entropy = Entropy + KL Divergence
H(P,Q)        = H(P)   + KL(P||Q)
```

What does this mean?

- **H(P)** is the entropy of the true distribution — the irreducible uncertainty in the data. No model can do better than this. It's a property of the problem, not the model.
- **KL(P||Q)** is the KL divergence — how much *extra* surprise the model adds beyond what's inherent in the data. This is the gap between your model and perfection.

**When training a model, H(P) is constant.** You can't change the data. So minimizing cross-entropy is equivalent to minimizing KL divergence — making your model's predictions match reality as closely as possible.

The minimum possible loss is H(P), achieved when Q = P (KL = 0).

> **Notebook Exercise 13.** Visualize the decomposition as a stacked bar chart.

---

## 8. KL Divergence

KL divergence (Kullback-Leibler divergence) measures how one probability distribution differs from another:

```
KL(P||Q) = ∑ P(x) log(P(x) / Q(x))
```

**Key properties:**

1. **Non-negative:** KL(P||Q) ≥ 0, always
2. **Zero iff equal:** KL(P||Q) = 0 if and only if P = Q
3. **NOT symmetric:** KL(P||Q) ≠ KL(Q||P) in general

The asymmetry matters:
- **Forward KL:** KL(P||Q) — penalizes when Q assigns low probability where P is high. The model must cover all modes of the data. This is what ML training minimizes.
- **Reverse KL:** KL(Q||P) — penalizes when Q assigns high probability where P is low. The model avoids putting mass where the data isn't.

> **Notebook Exercise 14.** Compute KL divergence and verify the decomposition.

> **Notebook Exercise 15.** See that KL(P||Q) ≠ KL(Q||P) with concrete examples.

---

## 9. Training = Minimizing Cross-Entropy

Now we can state precisely what "training" means:

1. You have training data drawn from some true distribution P
2. Your model produces a predicted distribution Q (parameterized by weights)
3. Training adjusts the weights to minimize H(P,Q)
4. Since H(P) is constant, this is equivalent to minimizing KL(P||Q)
5. The model converges when Q ≈ P

As training progresses:
- The predicted distribution Q moves closer to the true distribution P
- Cross-entropy decreases toward H(P)
- KL divergence decreases toward 0

> **Notebook Exercise 16.** Watch a model converge: Q approaches P over training steps.

> **Notebook Exercise 17.** Visualize the cross-entropy loss surface and see the unique minimum at Q = P.

---

## 10. Cross-Entropy in Practice

Let's connect this to a concrete example. Consider a bigram model predicting what character follows 'q':

- True distribution P: 'u' appears 99% of the time
- Early in training: model predicts uniformly → high cross-entropy
- Late in training: model predicts 'u' with high probability → low cross-entropy

The cross-entropy drops from ~1.6 nats (uniform) to ~0.06 nats (near-perfect prediction for this context).

In practice, we monitor training by plotting **loss curves** — cross-entropy over epochs:
- Training loss should decrease smoothly
- Validation loss should track training loss
- If validation >> training → overfitting
- If both plateau → model converged

> **Notebook Exercise 18.** See cross-entropy decrease as a bigram model learns.

> **Notebook Exercise 19.** Interpret training and validation loss curves.

---

## 11. Building Intuition

The best way to internalize cross-entropy is to experiment. Try different true and predicted distributions and observe how cross-entropy changes:

- **Confident and correct** → low cross-entropy
- **Uncertain (uniform)** → moderate cross-entropy
- **Confident and wrong** → very high cross-entropy!

Being confidently wrong is worse than being uncertain. This is why well-calibrated models (that know what they don't know) are valuable.

> **Notebook Exercise 20.** Experiment with your own distributions.

> **Notebook Exercise 21.** Visualize custom examples.

---

## Summary

**Entropy (Part 0):**
- **Entropy = unpredictability.** Three equivalent views: average yes/no questions, optimal compression size, average surprise.
- **Key properties:** non-negative, maximized by uniform distribution, conditioning reduces entropy.
- Measured in **bits** (log₂).

**Cross-Entropy and KL Divergence (Parts 1-4):**
- **Cross-entropy H(P,Q)** = average surprise of model Q when reality follows P.
- **Decomposition:** H(P,Q) = H(P) + KL(P||Q). Training minimizes KL(P||Q).
- **KL divergence** is NOT symmetric. Forward KL (what ML uses) ensures the model covers all modes.
- **Loss = cross-entropy** in nats (natural log). Lower loss = better model.
- **Training** = adjusting parameters to minimize cross-entropy = making Q match P.

**What's next (Session 3):**
- Apply these concepts to real data: Rock-Paper-Scissors and character bigrams
- Build transition matrices, measure their entropy, and exploit patterns
- Hit the V² wall that makes neural networks necessary

---

## Discussion Questions

1. You computed entropy three different ways (guessing game, compression, surprise). Which framing clicked for you? Which felt least intuitive?
2. Shannon's key insight was that information = reduction of uncertainty. How does this differ from everyday notions of "information" (like file size or word count)? When would these measures disagree?
3. Cross-entropy decomposes into H(P) + KL(P||Q). Why is it important that H(P) is constant during training? What would go wrong if it weren't?
4. KL divergence is not symmetric. In what practical situations would the choice between KL(P||Q) and KL(Q||P) lead to different model behavior?
5. Being "confidently wrong" produces higher cross-entropy than being "uncertain." How does this relate to model calibration? Why might a well-calibrated model be preferred over a more accurate but poorly calibrated one?
6. The loss function is just average surprise. Why does minimizing surprise produce a useful model? What are we assuming about the data?
7. We switched from bits (log₂) to nats (ln) for ML. Does the choice of log base affect what the model learns, or only the numerical scale of the loss?
8. In session 1, you saw that LLMs predict the next *token*. How would cross-entropy loss differ between a character-level model and a token-level model trained on the same text?
