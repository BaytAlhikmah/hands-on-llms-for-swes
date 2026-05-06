Two problems, one idea. You'll predict the next move in Rock-Paper-Scissors, then the next character in a name. Both use the same approach: count transitions, normalize to probabilities, exploit patterns. Then you'll see how a learning algorithm can discover those same probabilities automatically from data, without explicit counting.

Read this alongside `notebook.ipynb` — every hands-on section below maps to a cell you should run.

## Learning Objectives

By the end of this session, you will understand:

- What **entropy** is, in three different ways: average yes/no questions to guess an outcome, optimal compression size, and average surprise
- How to compute entropy from a probability distribution, and what "bits" actually measure
- That **predictable patterns have low entropy**, and exploiting those patterns is the core of both winning at games and building language models
- How to build a **transition matrix** by counting: if X just happened, what's the probability of Y next?
- That normalizing counts gives you a probability distribution, and you can sample from it to generate new sequences
- That a **learning algorithm** can discover the same probabilities as counting, by automatically finding patterns in the data
- Why transition matrices hit a wall when the vocabulary grows (the **V² problem**).

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

This isn't luck. With 8 equally likely options, you always need exactly **log₂(8) = 3** yes/no questions. That number — the average questions you need — is called **entropy**.

→ **Notebook Exercise 1.** Visualize the binary decision tree for the uniform case (1-8).

Now try a different game. Same setup, but this time I tell you: "50% of the time the number is 1. The other 50% is split equally among 2–8."

Your strategy changes:
- "Is it 1?" → 50% of the time: **done in 1 question**. 50% of the time: "No" → now you have 7 equally likely options left, which takes log₂(7) ≈ 2.807 more questions.

**Average questions needed:**
```
0.5 × 1  +  0.5 × (1 + 2.807)  =  2.4035 questions
```

→ **Notebook Exercise 2.** Visualize the binary decision tree for the biased case and compare with the uniform case.

→ **Notebook Exercise 3.** Visualize the probability distributions and see how entropy relates to the "spread" of probabilities.

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

→ **Notebook Exercise 4.** Compute entropy for different probability distributions using the formula.

---

**Key insight:** Lower entropy = more predictable = fewer questions needed.
- Uniform (all equal): 3.00 bits
- Biased (50% on one): 2.4035 bits
- Certain (100% on one): 0.00 bits

---

## 2. Entropy: The Formula

Here's the general formula. Don't memorize it — just notice what it does:

```python
import math

def entropy(probs: list[float]) -> float:
    """Average number of yes/no questions (bits) to guess the outcome."""
    return sum(p * math.log2(1/p) for p in probs if p > 0)

    # it's also written as
    # -sum(p * math.log2(p) for p in probs if p > 0)
    # Question: Why? How are both equivalent?
```

→ **Notebook Exercise 5.** Implement and test the entropy function on various distributions.

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

Entropy measures unpredictability. In the rest of this session, we'll use it to measure how predictable your friend's Rock-Paper-Scissors moves are, and how predictable English character sequences are.

---

## 3. Three Ways to Think About Entropy

Software engineers come from different backgrounds, so here are three equivalent framings. Pick the one that clicks for you — they're all the same math.

### A. Guessing Game (you just saw this)

Entropy = average number of yes/no questions to identify the outcome.

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

→ **Notebook Exercise 6.** Visualize how entropy-based compression saves space.

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

→ **Notebook Exercise 7.** Visualize the surprise curve and see how surprise relates to probability.

**This framing is critical for later:** when we train a model, we're minimizing its average surprise on real data. A well-trained model isn't surprised by what actually happens.

→ **Notebook Exercise 8.** Explore entropy across different probability distributions using interactive visualizations.

---

## 4. Hands-on: Rock-Paper-Scissors

Now that you understand entropy as a measure of unpredictability, let's apply it to a real problem.

Your friend thinks they play Rock-Paper-Scissors randomly. You've recorded 1,000 of their moves in `rps_opponent_moves.csv`. Let's see if they're actually random — and if their patterns have lower entropy than true randomness.

→ **Notebook Exercise 9.** Load the moves and check the overall frequencies.

**Predict first:** do the overall frequencies (% Rock, % Paper, % Scissors) tell you if the sequence is random?

**Answer:** No. Roughly 33% each looks random, but that doesn't mean the *sequence* is random. What matters is: **after Rock, what comes next?**

---

## 5. Transition Matrices

A **transition matrix** counts: "How often does Y follow X?"

For RPS, it's a 3×3 grid:

|           | → Rock | → Paper | → Scissors |
|-----------|--------|---------|------------|
| After Rock    | ?      | ?       | ?          |
| After Paper   | ?      | ?       | ?          |
| After Scissors| ?      | ?       | ?          |

→ **Notebook Exercise 10.** Count the transitions and fill the matrix.

You'll find something like:

| After Rock | → Paper 60% | → Rock 30% | → Scissors 10% |

**This is not random.** After Rock, your friend heavily favors Paper.

---

## 6. Entropy of Each Row

Each row of the transition matrix is a probability distribution over the next move. We can compute the entropy of each row using **the same formula from section 2** — but now applied to real data:

→ **Notebook Exercise 10 (continued).** Calculate entropy for each row.

```python
def row_entropy(probs: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)

# After Rock: [0.6, 0.3, 0.1]
print(f"After Rock: {row_entropy([0.6, 0.3, 0.1]):.3f} bits")

# Random player: [1/3, 1/3, 1/3]
print(f"Random: {row_entropy([1/3, 1/3, 1/3]):.3f} bits")  # 1.585
```

**Lower entropy = more predictable = easier to exploit.**

If your friend were truly random, every row would have entropy ≈ 1.585 bits. Instead, you'll see something closer to 1.2 bits after Rock — they're giving away information.

**This is Shannon's insight in action:** the entropy gap (1.585 - 1.2 = 0.385 bits) represents the information you can exploit to win more games.

---

## 7. Exploiting the Pattern

Now that you have the probabilities, you can predict their next move and play the counter:
- They'll likely play Paper → you play Scissors
- They'll likely play Scissors → you play Rock
- They'll likely play Rock → you play Paper

→ **Notebook Exercise 11.** Play against them using the counting model.

**Predict first:** what win rate would you expect against a random player? What about against your friend?

**Results:**
- Against random: **~33% wins, 33% losses, 33% draws** (no edge)
- Against your friend: **~50–60% wins** (you're reading them)

The entropy gap translates directly to win rate. **Prediction is exploitation.**

---

## 8. Learning the Same Thing Automatically

The counting approach works perfectly. But what if you *didn't* count explicitly? What if you started with a 3×3 matrix of **random numbers** and let an algorithm discover the patterns by looking at the data?

The helper class `TransitionLearner` does exactly this:
1. Starts with uniform probabilities (knows nothing)
2. Looks at each transition, measures how wrong its guess was
3. Adjusts the numbers to be slightly less wrong
4. Repeats until the loss stops improving

→ **Notebook Exercise 12.** Train a `TransitionLearner` on the RPS data.

You'll see the loss decrease over time. By the end, the learned probabilities are **nearly identical** to the counted ones.

**Key insight:** counting and learning from data solve the same problem. Counting works when the problem is tiny (3 moves, 1,000 examples). Learning works when the problem is huge (30,000 tokens, billions of examples) — you can't count every possible sequence, but you can learn patterns.

---

## 9. What Is "Loss"?

When the notebook prints "loss = 1.234", what does that number mean?

It's the **average surprise** across all transitions. More precisely, it's **cross-entropy**: the average `-log₂(probability the model assigned to what actually happened)`.

Example:
- Your model says: "After Rock, Paper has probability 0.4"
- Reality: your friend plays Paper
- Surprise: `-log₂(0.4) = 1.32 bits`

Do this for every transition and average. That's the loss.

**Lower loss = less surprised = better model.**

If the model were perfect (assigns probability 1.0 to what happens, 0 to everything else), loss would be 0. If the model is terrible (assigns probability 0.1 to what happens), loss is high.

This is what the learning algorithm minimizes. Every time it adjusts the numbers, it's trying to be less surprised by the data.

---

## 10. Cross-Entropy vs Entropy

Let's formalize:

```python
def cross_entropy(true_probs: list[float], model_probs: list[float]) -> float:
    """Average surprise if you use the wrong probabilities."""
    return sum(p_true * (-math.log2(p_model))
               for p_true, p_model in zip(true_probs, model_probs)
               if p_true > 0)
```

- **Entropy:** average surprise using the *true* distribution
- **Cross-entropy:** average surprise using your *model's* distribution

**Fact:** cross-entropy ≥ entropy. Equality only when your model matches reality perfectly.

Example:
```python
true = [0.6, 0.3, 0.1]  # After Rock, this is what happens
model_uniform = [1/3, 1/3, 1/3]  # Model thinks it's uniform
model_perfect = [0.6, 0.3, 0.1]  # Model is perfect

print(f"Entropy: {entropy(true):.3f}")  # 1.295
print(f"Cross-entropy (uniform model): {cross_entropy(true, model_uniform):.3f}")  # 1.585
print(f"Cross-entropy (perfect model): {cross_entropy(true, model_perfect):.3f}")  # 1.295
```

**Training = minimizing cross-entropy = getting closer to the true entropy.**

This is what's hidden inside `TransitionLearner`. It's measuring cross-entropy on every step and adjusting to lower it.

---

## 11. From 3×3 to 28×28: Character Bigrams

Same idea, bigger problem. Instead of 3 moves (Rock, Paper, Scissors), we have **28 characters**:
- 26 letters (a–z)
- `<S>` (start of name)
- `<E>` (end of name)

Instead of predicting the next RPS move, we predict the next character in a name.

The transition matrix grows from 3×3 to 28×28, but the approach is identical.

---

## 12. What Is a Bigram?

A **bigram** is a pair of consecutive characters.

Example: `"emma"` produces these bigrams:
```
<S> → e
e   → m
m   → m
m   → a
a   → <E>
```

The `<S>` and `<E>` tokens mark boundaries. Without them, the model wouldn't know which characters tend to start or end names.

→ **Notebook Exercise 13.** Print bigrams for a few names.

---

## 13. Counting Bigrams

→ **Notebook Exercise 14.** Build a 28×28 count matrix from the training data.

You'll see:
- Total bigrams: ~200,000
- Non-zero entries: ~500 / 784 (many pairs never occur)

Look at one row:
```
After 'm':
  m → 'a'  :  5,234 times
  m → 'e'  :  3,891 times
  m → 'i'  :  2,456 times
  m → <E>  :  1,823 times (names ending in 'm')
  ...
```

Normalize the counts to probabilities:
```
P('a' | 'm') = 5234 / (5234 + 3891 + ... ) ≈ 0.35
```

---

## 14. Entropy Per Character

Now compute the entropy of each row — "After seeing this character, how unpredictable is the next one?"

```python
# After 'q'
P_after_q = [0.0, 0.0, ..., 0.95, ...]  # almost always 'u'
print(f"After 'q': {entropy(P_after_q):.2f} bits")  # ~0.2

# After 'a'
P_after_a = [0.05, 0.08, 0.12, ...]  # many options
print(f"After 'a': {entropy(P_after_a):.2f} bits")  # ~3.5
```

**Average entropy across all characters ≈ 2.5 bits.** That's the average unpredictability per character in English names.

Compare to random:
```
Random character: log₂(28) = 4.81 bits
Bigram model: ~2.5 bits
```

**We've cut uncertainty in half** just by learning which characters follow which.

---

## 15. Generating Names

To generate a name:
1. Start with `<S>`
2. Sample the next character from `P(next | current)`
3. Repeat until we sample `<E>`

→ **Notebook Exercise 15.** Generate 20 names using the counting model.

**Predict first:** will they look like real names?

**Sample output:**
```
1. alia
2. kendra
3. jace
4. emira
5. xan
...
```

Some look plausible. Some are weird but pronounceable. None are random gibberish — the model learned *something* about English phonotactics just from counting pairs.

---

## 16. Measuring Quality: Loss

How do we measure "this model is better than that one"?

We compute **average negative log-likelihood** = **cross-entropy** on a test set:

```python
def compute_loss(P: list[list[float]], names: list[str]) -> float:
    total_surprise = 0.0
    n_bigrams = 0
    for name in names:
        chs = ['<S>'] + list(name) + ['<E>']
        for i in range(len(chs) - 1):
            curr = chs[i]
            next = chs[i + 1]
            prob = P[stoi[curr]][stoi[next]]
            total_surprise += -math.log2(prob) if prob > 0 else 20.0
            n_bigrams += 1
    return total_surprise / n_bigrams
```

→ **Notebook Exercise 16.** Compute train and test loss for the counting model.

**Results:**
```
Random guessing: 4.81 bits (uniform over 28 chars)
Bigram model (train): 2.45 bits
Bigram model (test):  2.47 bits
```

**Lower is better.** The model is much less surprised by real names than random guessing would be.

---

## 17. Learning Bigrams from Scratch

→ **Notebook Exercise 17.** Train a `TransitionLearner` on the bigram data.

Watch the loss decrease. After 200 steps, the learned probabilities are nearly identical to the counted ones:

```
Max absolute difference: 0.002
Train loss (learned): 2.45 bits
Test loss (learned):  2.47 bits
```

**Same answer, different path.** Counting works because we have enough data to see every bigram hundreds of times. The learning algorithm gets there by iteratively finding patterns that reduce surprise.

---

## 18. Generate from the Learned Model

→ **Notebook Exercise 18.** Generate names using the learned model.

**Predict first:** will they look different from the counting model's names?

**Answer:** No. The models are identical (within rounding error), so the outputs are the same.

---

## 19. The V² Wall

Our bigram model is a V × V matrix, where V = vocabulary size.

- V = 28 characters → 784 parameters
- V = 50,000 words → 2.5 billion parameters
- V = 128,000 tokens (Llama 3) → **16 billion parameters**

And you'd need trillions of examples to fill it. **This doesn't scale.**

→ **Notebook Exercise 19.** See the table of vocabulary sizes and memory requirements.

| Vocabulary | V | V² | Memory |
|------------|---|----|--------|
| Characters (this notebook) | 28 | 784 | 3 KB |
| GPT-2 BPE tokens | 50,257 | 2.5B | 10 GB |
| Llama 3 tokens | 128,256 | 16.5B | 66 GB |

**The problem is clear:** storing a full transition matrix for realistic vocabularies is computationally infeasible. Modern language models need a different approach to represent these probabilities efficiently.

**We'll explore better solutions in the next session.**

---

## Summary

- **Entropy measures unpredictability.** It's the average number of yes/no questions to guess an outcome, the optimal compression size, and the average surprise. All the same math.
- **Predictable sequences have low entropy.** Your friend's RPS moves had lower entropy than random (1.2 vs 1.585 bits), so you could exploit them.
- **A transition matrix counts "Y follows X"** and normalizes to probabilities. You can sample from it to generate new sequences.
- **Learning algorithms can discover the same probabilities as counting** by automatically finding patterns in the data and minimizing cross-entropy (average surprise).
- **Cross-entropy ≥ entropy.** Training a model = pushing cross-entropy down toward the true entropy.
- **Loss = cross-entropy = average surprise.** Lower loss = better model.
- **Character bigrams** are a 28×28 transition matrix. They cut per-character entropy from 4.8 bits (random) to 2.5 bits (bigram model).
- **Transition matrices hit a V² wall.** For large vocabularies (50K–128K tokens), storing a full V×V matrix is impossible. Modern language models use more efficient representations to solve this scaling problem.
- The counting model and the learned model produce identical results on this toy problem. But when the data is sparse and the vocabulary is huge, only learning works — and that's why neural language models exist.

---

## Discussion Questions

1. You computed entropy three different ways (guessing game, compression, surprise). Which framing clicked for you? Which felt least intuitive?
2. Shannon's key insight was that information = reduction of uncertainty. How does this differ from everyday notions of "information" (like file size or word count)? When would these measures disagree?
3. The RPS model had lower entropy than random, so you could win. Where else in software do you exploit predictability? (Think: caching, compression, autocomplete, prefetching...)
4. Cross-entropy is always ≥ entropy. When you trained the model, did the loss reach the true entropy? Why or why not?
5. We generated names by sampling from bigram probabilities. They looked plausible but weird. What information is the bigram model *missing* that a human has?
6. The V² wall shows that full transition matrices don't scale. What are the tradeoffs between having a complete V×V matrix versus using a more compressed representation?
7. The loss function is just average surprise. Why does minimizing surprise produce a useful model? What are we assuming about the data?
8. In session 1, you saw that LLMs predict the next *token*, not the next character. How would a token-level bigram model differ from a character-level one? Would the entropy be higher or lower?
