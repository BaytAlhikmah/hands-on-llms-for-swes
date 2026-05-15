In session 1, you saw that a language model does one thing: predict the next token. But how do you measure whether those predictions are any good? And how predictable is language in the first place?

In 1948, Claude Shannon answered both questions with a single idea. This session builds that idea from scratch — starting with entropy (the measure of unpredictability), then cross-entropy (the measure of prediction quality), and finally KL divergence (what training actually optimizes).

Read this alongside `notebook.ipynb` — every hands-on section below maps to a cell you should run.

## Learning Objectives

By the end of this session, you will understand:

- What **entropy** is, in three different ways: average yes/no questions to guess an outcome, optimal compression size, and average surprise
- How to compute entropy from a probability distribution, and what "bits" actually measure
- What **cross-entropy** is: the bits per symbol when you compress with the wrong distribution
- The decomposition **H(P,Q) = H(P) + KL(P||Q)**: optimal bits + wasted bits = actual bits
- What **KL divergence** measures (coding inefficiency) and why it's not symmetric
- The difference between **bits** (log₂) and **nats** (ln), and why ML uses nats
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

## 4. Cross-Entropy: Coding with the Wrong Distribution

Now we move from measuring the uncertainty *in the data* to measuring what happens when you *design a code for the wrong distribution*.

Suppose data comes from distribution P, but you design your compression scheme assuming distribution Q. What happens?

**Example: Morse Code for the Wrong Language**

Morse code was designed for English: 'E' gets the shortest code (·), 'T' gets second shortest (−), because they're the most common letters in English. This works great!

But what if you try to transmit French with English Morse code? French uses 'E' more than English does, but also uses accented letters (é, è, à) that English Morse doesn't efficiently encode. You're using a codebook designed for English (Q) to transmit French (P).

**Result:** You waste bits. Your messages are longer than necessary.

This waste is measured by **cross-entropy**:

```
H(P,Q) = -∑ P(x) log₂ Q(x)
```

It's the average number of bits needed per symbol when:
- Data comes from distribution P (reality)
- Your code was designed for distribution Q (your assumption)

**Key examples:**
- If Q = P (perfect code): H(P,Q) = H(P) — optimal compression
- If Q is uniform (generic code): H(P,Q) = log₂(n) — no compression
- If Q is very wrong: H(P,Q) is very high — terrible compression

**Lower cross-entropy = better compression = less waste.**

> **Notebook Exercise 9.** Visualize true vs assumed distributions and see how cross-entropy measures the coding inefficiency.

> **Notebook Exercise 10.** See which symbols contribute most to the wasted bits.

> **Notebook Exercise 11.** Implement cross-entropy from scratch.

---

## 5. A Concrete Example: Huffman Coding with Wrong Assumptions

Let's make this concrete with Huffman coding — an algorithm that builds optimal compression codes.

**Setup:** You're building a compression system for DNA sequences (A, C, G, T).

**Scenario 1: You have the right distribution**
- Your analysis says: A=40%, C=30%, G=20%, T=10%
- You build a Huffman code based on this
- Optimal code lengths: A=1 bit, C=2 bits, G=2 bits, T=3 bits
- Average bits per symbol: 0.4×1 + 0.3×2 + 0.2×2 + 0.1×3 = 1.7 bits
- This equals H(P) = 1.7 bits — perfect!

**Scenario 2: You assumed wrong**
- You *assumed* uniform: A=25%, C=25%, G=25%, T=25%
- You built a code with all symbols = 2 bits
- But actual data has A=40%, C=30%, G=20%, T=10%
- Average bits per symbol: 2.0 bits (since every symbol uses 2 bits)
- This equals H(P,Q) = 2.0 bits

**You wasted 0.3 bits per symbol** by using the wrong distribution!

The waste is exactly KL(P||Q), which we'll see next.

**Key insight:** Cross-entropy measures coding efficiency. If you know the true distribution P, you can compress to H(P) bits per symbol. If you design for Q instead, you use H(P,Q) bits per symbol — and the difference is wasted space.

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

In compression terms:

```
Actual bits used = Optimal bits + Wasted bits
H(P,Q)           = H(P)         + KL(P||Q)
```

What does each term mean?

- **H(P)** is the entropy of the true distribution — the theoretical minimum bits per symbol. No compression scheme can do better. This is a property of the data itself.
- **KL(P||Q)** is the KL divergence — the *extra* bits you waste per symbol by using the wrong code. This is the inefficiency introduced by using distribution Q when the reality is P.

**When you build a compression system, H(P) is fixed.** You can't change the data's entropy. So minimizing bits used (cross-entropy) is equivalent to minimizing wasted bits (KL divergence) — making your assumed distribution Q match reality P as closely as possible.

The best possible compression uses H(P) bits per symbol, achieved when Q = P (zero waste, KL = 0).

> **Notebook Exercise 13.** Visualize the decomposition as a stacked bar chart: optimal bits + wasted bits = total bits.

---

## 8. KL Divergence: The Wasted Bits

KL divergence (Kullback-Leibler divergence) measures the coding inefficiency — the extra bits you waste by using the wrong distribution:

```
KL(P||Q) = ∑ P(x) log₂(P(x) / Q(x))
```

Equivalently:
```
KL(P||Q) = H(P,Q) - H(P)
         = (bits you actually use) - (optimal bits)
         = wasted bits per symbol
```

**Key properties:**

1. **Non-negative:** KL(P||Q) ≥ 0, always. You can't use fewer bits than optimal.
2. **Zero iff equal:** KL(P||Q) = 0 if and only if P = Q. Zero waste only when your code is optimal.
3. **NOT symmetric:** KL(P||Q) ≠ KL(Q||P) in general.

**Why asymmetry?**

The asymmetry has a clear interpretation:

- **KL(P||Q):** You design a code for Q, but data comes from P. You waste bits when P has high probability but Q gave it a long code.
- **KL(Q||P):** You design a code for P, but data comes from Q. You waste bits when Q has high probability but P gave it a long code.

Different scenarios, different waste!

**Example:**
- True distribution P: A=80%, B=20%
- Wrong assumption Q: A=50%, B=50%
- Forward KL(P||Q): You designed a code treating A and B equally, but A appears 80% of the time. You gave A a longer code than needed. High waste!
- Reverse KL(Q||P): You designed a code optimized for A=80%, but data is actually balanced. Less problematic.

> **Notebook Exercise 14.** Compute KL divergence and verify the decomposition.

> **Notebook Exercise 15.** See that KL(P||Q) ≠ KL(Q||P) with concrete examples showing different waste patterns.

---

## 9. Adaptive Compression: Learning the Distribution

Now imagine you're building an *adaptive* compression system:

1. You observe data coming from some unknown distribution P
2. You build a code assuming distribution Q (which starts as a guess)
3. You measure how many bits you're using: H(P,Q)
4. You update Q to reduce the bits used
5. Since H(P) is constant (data's inherent entropy), reducing H(P,Q) means reducing KL(P||Q) — reducing wasted bits
6. Your code converges toward optimal when Q ≈ P

This is exactly what happens in **arithmetic coding** and **adaptive Huffman coding** — practical compression algorithms that learn the distribution on the fly!

As the compressor adapts:
- The assumed distribution Q moves closer to the true distribution P
- Bits per symbol decreases from H(P,Q) toward H(P)
- Wasted bits (KL divergence) decreases toward 0

> **Notebook Exercise 16.** Watch a code adapt: Q approaches P over time, bits decrease.

> **Notebook Exercise 17.** Visualize the cross-entropy surface and see the unique minimum at Q = P.

---

## 10. Compression in Practice: Context-Dependent Codes

Let's make this concrete with English text compression.

Consider compressing text where the previous character is 'q':

- True distribution P: 'u' appears 99% of the time after 'q'
- Naive code (uniform assumption): every letter gets ~5 bits → uses ~5 bits per character
- Smart code (learned distribution): 'u' gets 1 bit, rare letters get longer codes → uses ~1 bit per character

The compression ratio improves dramatically: from 5 bits to 1 bit!

This is exactly what **context-based compression** algorithms like PPM (Prediction by Partial Matching) do — they use different codes depending on context. After 'q', they use a code optimized for "mostly 'u'". After 't', they use a code optimized for "mostly 'h' or 'e'".

**Measuring compression quality:**
- Start with a generic code (uniform Q) → high bits per character (high H(P,Q))
- Learn from data, refine Q → fewer bits per character
- Best possible: Q = P → minimum bits (H(P))

The bit rate H(P,Q) tells you how well your compression is working!

> **Notebook Exercise 18.** See bits per character decrease as you refine distribution Q.

> **Notebook Exercise 19.** Compare bit rates for different compression schemes.

---

## 11. Building Intuition: Good Codes vs Bad Codes

The best way to internalize cross-entropy is to experiment with different codes. Try different assumed distributions Q and observe how compression efficiency changes:

- **Good match (Q ≈ P)** → low cross-entropy → efficient compression (few wasted bits)
- **Generic code (uniform Q)** → moderate cross-entropy → no compression (every symbol same length)
- **Very wrong (Q opposite of P)** → very high cross-entropy → terrible compression (short codes for rare symbols!)

Being very wrong is worse than being generic. A uniform code at least doesn't waste much. A badly mismatched code (short codes for rare symbols, long codes for common ones) can actually make files *larger* than uncompressed!

**Real-world example:** If you compressed Spanish text with a code optimized for Japanese character frequencies, you'd get massive files. A generic ASCII encoding would actually work better.

> **Notebook Exercise 20.** Experiment with different assumed distributions and see compression efficiency.

> **Notebook Exercise 21.** Visualize custom examples: optimal codes vs mismatched codes.

---

## Summary

**Entropy:**
- **Entropy = unpredictability.** Three equivalent views: average yes/no questions, optimal compression size, average surprise.
- **Key properties:** non-negative, maximized by uniform distribution, conditioning reduces entropy.
- Measured in **bits** (log₂).

**Cross-Entropy: Compression with Wrong Assumptions:**
- **Cross-entropy H(P,Q)** = average bits per symbol when you design a code for Q but data comes from P.
- If Q = P: optimal compression at H(P) bits per symbol.
- If Q ≠ P: you use H(P,Q) > H(P) bits per symbol — wasted space!

**KL Divergence: Measuring the Waste:**
- **Decomposition:** H(P,Q) = H(P) + KL(P||Q)
  - H(P) = optimal bits (data's inherent entropy)
  - KL(P||Q) = wasted bits (inefficiency from wrong distribution)
- **KL divergence** is NOT symmetric: KL(P||Q) ≠ KL(Q||P). Different mismatches waste bits differently.
- **Minimizing cross-entropy** = minimizing wasted bits = learning the true distribution.

**Connection to Machine Learning:**

You might be wondering: what does compression have to do with predicting the next token?

**Everything.** Prediction IS compression.

Think about it: If you can perfectly predict what comes next, you don't need to store it — just store the prediction. If you can't predict at all, you need to store every symbol fully. Good prediction = good compression.

Shannon showed that the best possible compression rate for a sequence equals its entropy — which equals the uncertainty in predicting the next symbol. The two problems are mathematically identical.

This is why language models minimize cross-entropy loss. When a model predicts the next token:
- The model outputs a distribution Q (probability over next tokens)
- Reality reveals the actual next token (drawn from distribution P)
- Cross-entropy H(P,Q) measures the "bits per token" if you compressed with Q but reality is P
- Training minimizes cross-entropy = minimizes bits needed = maximizes prediction quality

**Prediction quality = Compression efficiency = Low cross-entropy.** Same thing, three names.

In Session 3, you'll see this connection in action: building models that predict (and implicitly compress) real sequences.

**What's next (Session 3):**
- Apply these concepts to real data: Rock-Paper-Scissors and character bigrams
- Build transition matrices, measure their entropy, and exploit patterns
- Hit the V² wall that makes neural networks necessary

---

## Discussion Questions

1. You computed entropy three different ways (guessing game, compression, surprise). Which framing clicked for you? Which felt least intuitive?
2. Shannon's key insight was that information = reduction of uncertainty. How does this differ from everyday notions of "information" (like file size or word count)? When would these measures disagree?
3. Cross-entropy decomposes into H(P) + KL(P||Q). In compression terms: optimal bits + wasted bits = actual bits. Why is H(P) fixed? What determines it?
4. KL divergence is not symmetric. Can you think of a practical example where compressing P-data with a Q-code wastes different amounts than compressing Q-data with a P-code?
5. Being "very wrong" (Q opposite of P) produces higher cross-entropy than being "generic" (uniform Q). Why is a mismatched code worse than no compression at all? When would you prefer a generic code over a learned one?
6. Morse code assigns short codes to frequent English letters. What would happen if you used Morse code to transmit: (a) German text, (b) DNA sequences, (c) random bits? Estimate whether cross-entropy would be higher or lower than optimal.
7. We switched from bits (log₂) to nats (ln) for ML. Does the choice of log base affect compression or prediction, or only the numerical scale?
8. The summary claims "prediction IS compression." Can you explain this connection? Why does a model that's good at predicting the next token also achieve good compression?
9. In session 1, you saw that LLMs predict the next *token*. How would the entropy (bits per symbol) differ between a character-level model and a token-level model trained on the same text? Which would have lower entropy per *position*? Why?
