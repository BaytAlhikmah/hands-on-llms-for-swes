In chapter 1, you saw that a language model does one thing: predict the next token. But how do you measure whether those predictions are any good? And how predictable is language in the first place?

In 1948, Claude Shannon answered both questions with a single idea. This chapter builds that idea from scratch — starting with entropy (the measure of unpredictability), then cross-entropy (the measure of prediction quality).

Read this alongside `notebook.ipynb` — the exercises below correspond directly to numbered exercises in the notebook.

## Learning Objectives

By the end of this chapter, you will understand:

- What **entropy** is, in three different ways: average yes/no questions to guess an outcome, optimal compression size, and average surprise
- How to compute entropy from a probability distribution, and what "bits" actually measure
- What **cross-entropy** is: the bits per symbol when you compress with the wrong distribution
- The difference between **bits** ($\log_2$) and **nats** ($\ln$), and why ML uses nats
- Why **prediction IS compression**, connecting information theory to machine learning

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

This isn't luck. With 8 equally likely options, an optimal strategy needs exactly $\log_2(8) = 3$ yes/no questions on average — which matches the **entropy** of the distribution.

> **Notebook Exercise 1.** Visualize the binary decision tree for the uniform case (1-8).

Now try a different game. Same setup, but this time I tell you: "50% of the time the number is 1. The other 50% is split equally among 2–8."

Your strategy changes:
- "Is it 1?" → 50% of the time: **done in 1 question**. 50% of the time: "No" → now you have 7 equally likely options left, which takes $\log_2(7) \approx 2.807$ more questions.

**Average questions needed:**
$$0.5 \times 1 + 0.5 \times (1 + 2.807) = 2.4035 \text{ questions}$$

> **Notebook Exercise 2.** Visualize the binary decision tree for the biased case and compare with the uniform case.

> **Notebook Exercise 3.** Visualize the probability distributions and see how entropy relates to the "spread" of probabilities.

---

### Why "2.4035 questions"? Don't questions give yes/no answers?

Yes! But we're computing the **expected value** (average across many games). Sometimes you get lucky (1 question), sometimes you need 4. On average: 2.4035 questions.

---

### The Same Calculation, Three Ways

**Method 1: Grouped by outcome** (what we just did)
$$0.5 \times 1 + 0.5 \times (1 + 2.807) = 2.4035$$

**Method 2: Each number individually**

Instead of grouping "2–8" together, let's count each one:
- Number 1: probability 0.5, needs 1 question → $0.5 \times 1 = 0.500$
- Numbers 2–8: probability 0.071 each, needs 3.807 questions → $7 \times (0.071 \times 3.807) = 1.903$
- Total: $0.5 + 1.903 = 2.4035$ questions

**Method 3: The entropy formula**

Each number has probability $P$ and needs $\log_2(1/P)$ yes/no questions. Substituting:

- Number 1: $P = 0.5$, so $\log_2(1/0.5) = \log_2(2) = 1$
- Numbers 2–8: total $P = 0.5$ (split equally), so $\log_2(1 / (0.5/7)) = \log_2(14) \approx 3.807$

$$0.5 \times \log_2(2) + 7 \times (0.5/7) \times \log_2(14) = 0.5 \times 1 + 0.5 \times 3.807 = 2.4035 \text{ bits}$$

This is the general pattern:
$$\text{Entropy} = \sum P(x) \times \log_2(1/P(x)) = 2.4035 \text{ bits}$$

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

$$H(X) = \sum_{x} P(x) \log_2\left(\frac{1}{P(x)}\right) = -\sum_{x} P(x) \log_2 P(x)$$

**Why "bits"?** A **bit** means one binary decision — one yes/no distinction. Because each yes/no question has two possible outcomes, entropy measured with $\log_2$ naturally counts information in bits.

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

Entropy of $[0.6, 0.2, 0.1, 0.1]$ = **1.57 bits** — our encoding is near-optimal!

> **Notebook Exercise 6.** Visualize how entropy-based compression saves space.

**Key insight:** if a sequence has low entropy, you can compress it well. If it has high entropy, it's already random-looking and incompressible.

### C. Surprise

Entropy = average surprise when you see the outcome.

When something with probability $p$ happens, your **surprise** is:
$$\text{surprise} = \log_2(1 / p) = -\log_2(p)$$

| Event | Probability | Surprise (bits) |
|-------|------------|-----------------|
| Sun rises | 0.99 | $\log_2(1 / 0.99) \approx 0.01449$ |
| Coin flip | 0.50 | $\log_2(1 / 0.50) = 1.00$ |
| Win lottery | 0.000001 | $\log_2(1 / 0.000001) \approx 19.9315$ |

Entropy is just the **expected surprise** — the average over all possible outcomes weighted by their probabilities.

> **Notebook Exercise 7.** Visualize the surprise curve and see how surprise relates to probability.

**This framing is critical for what follows:** cross-entropy measures the average surprise of a *model* on real data. A well-trained model isn't surprised by what actually happens.

> **Notebook Exercise 8.** Explore entropy across different probability distributions using interactive visualizations (binary entropy, ternary entropy, and mathematical properties).

---

### Properties of Entropy

Now that you've seen entropy in action, here are its key mathematical properties. These aren't just formalities — they capture fundamental facts about information and uncertainty.

**1. Non-negativity**
$$H(X) \geq 0$$
Entropy is always non-negative. It equals zero only when one outcome has probability 1 (complete certainty).

**2. Maximum Entropy**
$$H(X) \leq \log_2(n)$$
For a discrete random variable with $n$ possible outcomes, entropy is maximized when all outcomes are equally likely (uniform distribution). Maximum $= \log_2(n)$ bits.

**3. Continuity**

$H$ is a continuous function of the probability distribution. Small changes in probabilities lead to small changes in entropy.

**4. Symmetry**

$H$ is invariant under permutation of outcomes. The order doesn't matter, only the probabilities.

**5. Additivity (for independent events)**
$$H(X,Y) = H(X) + H(Y) \quad \text{[if X and Y are independent]}$$

The joint entropy of two independent random variables equals the sum of their individual entropies.

**6. Chain Rule**
$$H(X,Y) = H(X) + H(Y|X)$$

Joint entropy equals the entropy of $X$ plus the conditional entropy of $Y$ given $X$.

**7. Conditioning Reduces Entropy**
$$H(Y|X) \leq H(Y)$$

Knowing $X$ can only reduce (or leave unchanged) uncertainty about $Y$. Information never increases uncertainty. Equality holds when $X$ and $Y$ are independent.

> **Notebook Exercise 8c.** See these properties demonstrated with concrete examples.

---

## 4. Cross-Entropy: Coding with the Wrong Distribution

> **See Notebook Part 2** — this section corresponds to Exercises 9-15 in the notebook.

Now we move from measuring the uncertainty *in the data* to measuring what happens when you *design a code for the wrong distribution*.

**The Setup:**

Suppose you're compressing a file: `AAAAAABBCD` (10 characters)

**True distribution:** $P = \{A: 60\%, B: 20\%, C: 10\%, D: 10\%\}$

But what if you design your code assuming a different distribution $Q$? What happens?

**The Key Idea:**

This waste is measured by **cross-entropy**:

$$H(P,Q) = -\sum_{x} P(x) \log_2 Q(x)$$

It's the average number of bits needed per symbol when:
- Data comes from distribution $P$ (reality)
- Your code was designed for distribution $Q$ (your assumption)

**Lower cross-entropy = better compression = less waste.**

Let's see this in action with four different codebooks:

**Data distribution:** $P = \{A: 60\%, B: 20\%, C: 10\%, D: 10\%\}$

---

### Scenario 1: Optimal Codebook (Q = P)

> **Notebook Exercise 9.** Tree 1 - Optimal for true distribution P

You know the true distribution and design an optimal Huffman code:
- A (60%): `0` (1 bit)
- B (20%): `10` (2 bits)
- C (10%): `110` (3 bits)
- D (10%): `111` (3 bits)

**Average bits:** $0.6 \times 1 + 0.2 \times 2 + 0.1 \times 3 + 0.1 \times 3 = 1.6$ bits

This approximates $H(P) \approx 1.57$ bits — near-optimal!

### Scenario 2: Uniform Codebook

> **Notebook Exercise 10.** Tree 2 - Uniform codebook

You assume all symbols are equally likely: $Q = \{A: 25\%, B: 25\%, C: 25\%, D: 25\%\}$

All symbols get 2-bit codes: A=`00`, B=`01`, C=`10`, D=`11`

**Average bits:** $0.6 \times 2 + 0.2 \times 2 + 0.1 \times 2 + 0.1 \times 2 = 2.0$ bits

**Wasted:** $2.0 - 1.6 = 0.4$ bits per symbol (+25% worse than optimal)

### Scenario 3: Wrong-Biased Codebook

> **Notebook Exercise 11.** Tree 3 - Wrong-biased codebook

You wrongly assume C is most frequent: $Q = \{C: 50\%, A: 25\%, B: 12.5\%, D: 12.5\%\}$

But in reality, A is most frequent! Your code wastes bits:
- C gets `0` (1 bit) but only appears 10% of the time
- A gets `10` (2 bits) but appears 60% of the time

**Average bits:** $0.6 \times 2 + 0.2 \times 3 + 0.1 \times 1 + 0.1 \times 3 = 2.2$ bits

**Wasted:** $2.2 - 1.6 = 0.6$ bits per symbol (+38% worse)

### Scenario 4: Reversed-Priority

> **Notebook Exercise 12.** Tree 4 - Reversed-priority codebook

The complete opposite of optimal: $Q = \{D: 50\%, C: 25\%, A: 12.5\%, B: 12.5\%\}$

- D (10% actual) gets 1 bit ← shortest code for **least frequent**!
- A (60% actual) gets 3 bits ← longest code for **MOST frequent**!

**Average bits:** $0.6 \times 3 + 0.2 \times 3 + 0.1 \times 2 + 0.1 \times 1 = 2.7$ bits

**Wasted:** $2.7 - 1.6 = 1.1$ bits per symbol (+69% worse!)

> **Notebook Exercise 13.** Side-by-side comparison of all four trees

> **Notebook Exercise 14.** Calculate entropy and cross-entropy numerically for all scenarios

> **Notebook Exercise 15.** Interactive exploration of the full coding spectrum

**Key insight:** Cross-entropy measures coding efficiency. The more your assumed distribution $Q$ differs from reality $P$, the more bits you waste. Being very wrong is worse than being generic (uniform).

---

## 5. Bits vs Nats

> **See Notebook Part 3** — this section corresponds to Exercise 16 in the notebook.

So far, we measured entropy in **bits** using $\log_2$. In machine learning, we switch to **nats** using natural log ($\ln$).

Why?
- PyTorch's `nn.CrossEntropyLoss()` uses natural log
- Derivatives are cleaner: $\frac{d}{dx}(\ln x) = \frac{1}{x}$ (vs $\frac{d}{dx}(\log_2 x) = \frac{1}{x \ln 2}$)
- Standard in ML papers and frameworks

**Conversion:** $\text{bits} = \frac{\text{nats}}{\ln(2)} \approx \text{nats} \times 1.443$

From now on, when we write cross-entropy and loss, we use **nats** (natural log) unless stated otherwise.

> **Notebook Exercise 16.** Compare cross-entropy in bits vs nats.

---

## Summary

**Entropy:**
- **Entropy = unpredictability.** Three equivalent views: average yes/no questions, optimal compression size, average surprise.
- **Key properties:** non-negative, maximized by uniform distribution, conditioning reduces entropy.
- Measured in **bits** ($\log_2$) or **nats** ($\ln$).

**Cross-Entropy: Compression with Wrong Assumptions:**
- **Cross-entropy** $H(P,Q)$ = average bits per symbol when you design a code for $Q$ but data comes from $P$.
- If $Q = P$: optimal compression at $H(P)$ bits per symbol.
- If $Q \neq P$: you use $H(P,Q) > H(P)$ bits per symbol — wasted space!
- Being very wrong (opposite distribution) wastes more bits than being generic (uniform).

**Connection to Machine Learning:**

You might be wondering: what does compression have to do with predicting the next token?

**Everything.** Prediction IS compression.

Think about it: If you can perfectly predict what comes next, you don't need to store it — just store the prediction. If you can't predict at all, you need to store every symbol fully. Good prediction = good compression.

Shannon showed that the best possible compression rate for a sequence equals its entropy — which equals the uncertainty in predicting the next symbol. The two problems are mathematically identical.

This is why language models minimize cross-entropy loss. When a model predicts the next token:
- The model outputs a distribution $Q$ (probability over next tokens)
- Reality reveals the actual next token (drawn from distribution $P$)
- Cross-entropy $H(P,Q)$ measures the "bits per token" if you compressed with $Q$ but reality is $P$
- Training minimizes cross-entropy = minimizes bits needed = maximizes prediction quality

**Prediction quality = Compression efficiency = Low cross-entropy.** Same thing, three names.

In Chapter 3, you'll see this connection in action: building models that predict (and implicitly compress) real sequences.

---

## Discussion Questions

1. You computed entropy three different ways (guessing game, compression, surprise). Which framing clicked for you? Which felt least intuitive?
2. In Exercise 14, we calculated that the uniform codebook uses 2.0 bits per character while optimal uses 1.6 bits. Where did those 0.4 wasted bits come from? What specific symbols contributed most to the waste?
3. Being "very wrong" ($Q$ opposite of $P$) produces higher cross-entropy than being "generic" (uniform $Q$). Why is a mismatched code worse than no compression at all? When would you prefer a generic code over a learned one?
4. The four-tree comparison shows cross-entropy ranging from 1.6 to 2.7 bits. But the notebook mentions cross-entropy can approach infinity. How? What distribution $Q$ would produce infinite cross-entropy for our data $P = \{A: 60\%, B: 20\%, C: 10\%, D: 10\%\}$?
5. We switched from bits ($\log_2$) to nats ($\ln$) for ML. Does the choice of log base affect compression or prediction, or only the numerical scale?

