**Note:** This chapter is heavily inspired by Andrej Karpathy's excellent ["The spelled-out intro to language modeling: building makemore"](https://www.youtube.com/watch?v=PaCmpygFfXo) lecture. We highly recommend watching it alongside this material — particularly the bigram model section (first ~45 minutes).

---

In Chapter 2, you learned to measure uncertainty (entropy) and prediction quality (cross-entropy). Those were theoretical tools. Now we use them on real data.

This chapter takes you from a 3×3 game (Rock-Paper-Scissors) to a 28×28 character model, showing how counting works beautifully at small scale — then breaks. You'll see why neural networks became necessary, not by being told, but by hitting the wall yourself.

You'll also encounter your first **learning algorithm** — a black box that automatically finds good probabilities from data. We won't open that box yet (that's Chapter 4), but you'll use it and see that it works. Two paths to the same answer: one explicit (counting), one automatic (learning).

Read this alongside `notebook.ipynb` — every section below maps to numbered exercises you should run.

---

## 0. The Moment Statistical Models Hit a Wall

In the 1980s and 90s, statistical language models dominated NLP. Researchers at IBM, including Frederick Jelinek and Robert Mercer, built systems based on **n-grams**, models that estimate the probability of the next word from how often word sequences appear in large corpora.

A **bigram model** estimates how likely word B is to follow word A. A **trigram** uses the previous two words. These systems were fundamentally **statistical**: they relied on counting observed sequences and converting those counts into probabilities, often with sophisticated smoothing techniques (Katz backoff, Good-Turing, Kneser-Ney) and pruning to handle missing data.

But they faced a fundamental limitation: **data sparsity**. The number of possible n-grams grows explosively with vocabulary size:

- Characters (27 symbols): **27² = 729** possible bigrams. Manageable.
- English words (50,000): **50,000² = 2.5 billion** possible bigrams. Most never observed.
- Modern token vocabularies (100,000–200,000): **10–40 billion** possible pairs. Extremely sparse.

Real systems used sparse storage (tries, compressed suffix structures, pruning), not dense tables. The problem wasn't literally storing every pair — it was that **most combinations were never seen**, and the models couldn't generalize to unseen word pairs.

This is the **data sparsity problem**. An n-gram model sees "black cat" and "white cat" as completely unrelated counts. It can memorize frequent patterns but struggles to generalize to rare or unseen combinations. As vocabularies and context lengths grew, simply counting sequences stopped scaling effectively.

In this chapter, you'll build a working bigram model from counts, generate plausible names from it, and hit the sparsity wall yourself. The numbers will show you **why counting eventually fails** — and why something fundamentally different became necessary.

---

## Learning Objectives

By the end of this chapter, you will understand:

- How to build a **transition matrix** by counting pairs in a sequence
- How to compute **row entropy** to measure predictability given context
- The difference between **counting** (explicit frequency table) and **learning** (automatic method that arrives at the same result)
- How to **generate** sequences by sampling from a learned probability distribution
- Why **data sparsity** makes explicit counting fail to generalize, and why something fundamentally different became necessary
- That a simple bigram model can produce surprisingly plausible text — and where it breaks

---

## Part 0: Setup

→ **Notebook Exercise 0.** Install the course package.

We'll use a helper class called `TransitionLearner` — it learns transition probabilities automatically by adjusting parameters from data. **Treat it as a black box for now.** You don't need to know how it works yet; you only need to know that **on fully-observed problems**, it converges to the same probabilities as counting. Chapter 4 opens the box and shows you what's inside.

---

## Part 1: Rock-Paper-Scissors (3×3 Transition Matrix)

You have 1,000 recorded moves from an opponent who claims to play randomly. Let's test that claim.

### Exercise 1: Load RPS Moves

→ **Notebook Exercise 1.** Load the moves, check overall frequencies.

The overall distribution looks roughly uniform: ~33% Rock, ~33% Paper, ~33% Scissors. If you only looked at these marginal frequencies, you'd think the opponent is random.

But **marginal entropy** (ignoring context) isn't the full story. Does knowing the *previous* move help predict the *next* one?

---

### Exercise 2: Count Transitions and Calculate Entropy

→ **Notebook Exercise 2.** Build a 3×3 transition count matrix: entry [i, j] = how many times move j followed move i.

Print the matrix as a table and visualize it as a heatmap.

**Predict first:** After your opponent plays Rock, are they equally likely to play Rock, Paper, or Scissors next? Or is there a pattern?

---

#### Compute row entropy

Now calculate the entropy of each row (each row is a conditional distribution: P(next | previous)).

→ **Notebook Exercise 2 (continued).** For each previous move, compute the entropy of the next-move distribution.

Compare to the uniform baseline (1/3, 1/3, 1/3) which has entropy = log₂(3) ≈ 1.585 bits.

**Key insight:** Lower entropy = more predictable. If the row entropy is below 1.585 bits, knowing the previous move gives you information. The entropy gap represents exploitable patterns.

**Optional deep dive:** To see how row entropies combine into conditional entropy H(Y|X), refer to the [supplemental notebook `entropy_from_counts.ipynb`](https://github.com/BaytAlhikmah/hands-on-llms-for-swes/blob/main/chapters/3/supplemental/entropy_from_counts.ipynb). It walks through the full derivation: marginal entropy H(Y) vs conditional entropy H(Y|X), and proves that H(Y|X) ≤ H(Y) — context reduces uncertainty.

---

### Exercise 3: Exploit the Patterns

If we can predict their next move better than random, we play the counter:
- They play Rock → we play Paper
- They play Paper → we play Scissors
- They play Scissors → we play Rock

→ **Notebook Exercise 3.** Implement a random baseline strategy (win rate ~33%), then implement a strategy that uses the transition matrix to predict their most likely next move and plays the counter.

**Predict first:** How much better than random can you do?

---

### Exercise 4: Learn the Probabilities Automatically

So far, you **counted** transitions explicitly. Now let's **learn** them.

We'll use `TransitionLearner`: it starts from uniform probabilities (knows nothing), looks at the data, and adjusts its probabilities automatically. After 200 steps, it converges to nearly the same probabilities you got by counting (very close but not exactly identical).

**How does it do this?** That's the mystery we'll unpack in Chapter 4. For now, just observe: **on this small, fully-observed problem**, two different methods reach the same solution.

→ **Notebook Exercise 4.** Train a `TransitionLearner` on the RPS transitions, visualize the learned probability matrix, and compare it to the counted matrix.

**Key insight:** Counting works perfectly here because the problem is tiny (3 moves, 9 transitions, ~300 samples per cell). Learning converges to the same solution.

But what if you couldn't see every transition hundreds of times? In **sparse settings**, learning can generalize beyond observed counts — smoothing, interpolating, sharing statistical strength. That's where learning becomes necessary, not just equivalent.

---

## Part 2: Character Bigrams (28×28 Transition Matrix)

Same idea, bigger problem. Instead of 3 moves, we have 28 characters (a–z plus `<S>` for start and `<E>` for end). Instead of predicting the next RPS move, we predict the next character in a name.

The transition matrix grows from 3×3 to 28×28 — but the approach is identical.

---

### What is a bigram?

A **bigram** is a pair of consecutive characters. The name "emma" produces:

```
<S> -> e    (start → 'e')
e   -> m
m   -> m
m   -> a
a   -> <E>  ('a' → end)
```

The `<S>` and `<E>` tokens mark where names begin and end. Without them, the model wouldn't know which characters tend to start or finish names.

---

### Exercise 5: Load Names Dataset

→ **Notebook Exercise 5.** Load ~32,000 names, split 90/10 into train/test, build a vocabulary of 28 tokens.

Each name becomes a sequence of bigrams. The name "emma" has 5 bigrams. A 10-character name has 11 bigrams (including `<S>` and `<E>`).

---

### Exercise 6: Build 28×28 Count Matrix

→ **Notebook Exercise 6.** Build the count matrix from the **training set only**: entry [i, j] = how many times character j followed character i.

Print statistics:
- Total bigram occurrences
- How many of the 28×28 = 784 entries are non-zero?
- What follows 'm'? (Print the top 10)

Then normalize each row to probabilities: P(next | current).

Visualize both the count matrix and the probability matrix as heatmaps.

**Predict first:** What patterns do you expect? Will vowels follow consonants more than consonants follow consonants?

---

### Exercise 7: Generate Names

To generate a name:
1. Start with `<S>`
2. Sample the next character from P(next | current)
3. Repeat until you sample `<E>`

→ **Notebook Exercise 7.** Generate 20 names from the counted bigram probabilities.

**Predict first:** Will these names look real?

---

### Exercise 8: Learn Bigrams with TransitionLearner

Same as RPS: use `TransitionLearner` to learn the 28×28 probability matrix from scratch. Still a black box, still mysterious.

→ **Notebook Exercise 8.** Prepare training pairs (every bigram as (input_id, target_id)), train for 200 steps, compare the learned matrix to the counted matrix.

**Key question:** How close are they? Check the max absolute difference across all 784 entries.

---

### Exercise 9: Generate from Learned Model

→ **Notebook Exercise 9.** Generate 20 names from the learned model.

**Predict first:** Will these names look different from the counting model's names?

They shouldn't — **on this fully-observed 28×28 problem**, the learned model converged to nearly the same probabilities as counting (check the max absolute difference in Exercise 8). Two paths, nearly the same destination.

But this near-equivalence only holds when every transition is well-observed. In sparse, high-dimensional settings (50K+ vocabulary), learning would generalize differently than pure counting.

---

### Exercise 10: The Data Sparsity Problem

Our bigram model is a V × V matrix. With V = 28 characters, that's 784 entries. Tiny.

But what if we used **words** instead of characters?

**Note:** Unlike previous exercises, we won't run a learned version here. The point of this exercise is to see where counting breaks — the numbers speak for themselves.

→ **Notebook Exercise 10.** Print a table showing vocabulary size, V², and memory requirements for:
- Characters (this notebook): 28
- GPT-2 tokens: 50,257
- Typical English words: 100,000
- Llama 3 tokens: 128,256
- Arabic news corpus: 300,000

**The numbers:**
- 28 characters → 784 entries → **3 KB**
- 50K tokens → 2.5 billion entries → **10 GB**
- 128K tokens → 16 billion entries → **65 GB**
- 300K words → 90 billion entries → **360 GB**

And this is for **bigrams only** (context length = 1). Trigrams (context = 2) multiply by V again.

**Key insight:** The problem isn't just storage (though that's real). The deeper issue is **generalization**.

Even with sparse storage, most word pairs are never observed. An n-gram model can't generalize from "black cat" to "white cat" — it treats every pair independently. If you've never seen "purple cat" in training, the model has no basis to predict it, even if you've seen "purple dog" and "black cat" thousands of times.

Explicit counting eventually hits its limits: it can't generalize beyond observed statistics. You need a way to share statistical strength across similar patterns. But how?

That question — and its answer — comes in later chapters.

But there's another fundamental limitation: **context length**. Even if storage were free and every bigram were observed, a bigram model only remembers one previous token. A trigram remembers two. Language often depends on context dozens or hundreds of words earlier — long-range syntax, semantic coherence, anaphora. N-gram models are fundamentally local; they can't track dependencies beyond their fixed window. Addressing this required not just learned representations, but architectures that could handle arbitrarily long contexts. That's where transformers come in — but that's a story for later chapters.

---

## Summary

**From Counting to Learning:**

- A **transition matrix** counts how often Y follows X, then normalizes to probabilities P(Y|X).
- **Row entropy** measures how predictable each context is. Lower entropy = more exploitable patterns.

- **Counting** works perfectly when the problem is small and every transition is observed many times.

- **Learning** converges to nearly the same probabilities as counting **in fully-observed settings** (like our 3×3 and 28×28 examples) — very close but not bit-identical due to the iterative optimization process. But in sparse, high-dimensional settings, learning can generalize beyond observed counts through parameter sharing and interpolation. *How* it does this is Chapter 4's question.

**Rock-Paper-Scissors (3×3):**
- Opponent's moves looked random (marginal entropy ≈ uniform), but transitions revealed strong patterns (conditional entropy < marginal).
- Exploiting those patterns significantly boosted win rate above the random baseline (the exact improvement depends on how non-random the recorded moves are).
- Both counting and learning produced nearly identical strategies.

**Character Bigrams (28×28):**
- Same approach, larger vocabulary. 784 entries, all observable in training.
- Generated plausible names: "kaleigh", "jaivion", "adelynn" — because English character patterns are predictable.
- Counting and learning again converged to the same model.

**The Data Sparsity Wall:**
- At 28 characters, explicit counting works. At 50K–300K tokens, it breaks:
  - 50K tokens → 2.5 billion possible bigrams
  - 128K tokens → 16 billion possible pairs
  - Most are **never seen** in training
- The problem isn't just storage (sparse data structures help). It's **generalization**: n-grams treat every pair independently and can't share statistical strength across similar words.
- This fundamental limitation drove the search for different approaches. Those approaches are the subject of later chapters.

**What's next:** In this chapter, `TransitionLearner` was a black box. You used it, but you didn't see inside. Chapter 4 opens that box.

You'll learn:
- What a **neuron** actually is (spoiler: it's just `y = w×x + b`)
- How a computer figures out which direction to move the weights
- Why **activation functions** matter (sigmoid, ReLU)
- What happens when one neuron isn't enough (the XOR problem)
- How multiple neurons working together can learn patterns a single neuron can't

---

## Discussion Questions

1. **Counting vs Learning:** The RPS and bigram models converged to nearly identical probabilities via counting and learning. If they're equivalent, why bother learning? When does learning become necessary? And more importantly: **how does the learning actually work?** What's happening inside that black box during those 200 training steps?

2. **Row Entropy as a Predictor:** We computed the entropy of each row (e.g., "after Rock, what's next?"). A row with entropy 0.8 bits is more predictable than one with 1.5 bits. How could you use row entropies to decide **when** to exploit vs play randomly?

3. **Generated Names:** The bigram model produces names like "kaleigh" and "maren" — which look plausible — but also "nglen" and "kaleigh" (repeated). What's missing? What patterns can a bigram model (context length = 1) **never** capture?

4. **Unseen Character Combinations:** In the 28×28 character bigram model, suppose you've seen "ma" 1000 times (in "maren", "mary", etc.) and "ba" 500 times (in "barbara", "bailey"), but never "za". Your training data has plenty of evidence that vowel 'a' commonly follows consonants. Yet the model assigns P('a'|'z') based purely on the (possibly zero) count of "za" — it can't leverage what it learned from "ma" and "ba". What would the model need to share statistical strength across similar patterns? Why does explicit counting treat 'm', 'b', and 'z' as completely independent?

5. **Sparse Storage vs Sparse Data:** Real n-gram systems used sparse storage (tries, pruning) to avoid storing zeros. So why didn't that solve the problem? What's the difference between "efficient storage of observed counts" and "generalizing to unseen combinations"? What would you need beyond efficient storage to make predictions about pairs you've never seen?

6. **From Bigrams to Trigrams:** If we extended to **trigrams** (context = 2 characters), the table becomes V³. For V = 28, that's 21,952 entries. For V = 50,000, it's **125 trillion entries**. At what vocabulary size does even a *sparse* trigram table become infeasible?

