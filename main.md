# Weight Tying

## What it is

**Weight tying** means reusing the **same set of weights in more than one place** in a model. Instead of having independent matrices, two (or more) parts of the model point to a **single tensor**.

In practice:

- **Fewer parameters**: the shared tensor is counted once.
- **Regularization**: the same weights must serve multiple purposes, which reduces overfitting.
- **Summed gradients**: since it is the same tensor, the gradients from all uses are added together in it.

> It is not "borrowing" weights from one layer to another. There is **a single tensor** in memory, and several parts of the model use that tensor.

The three most famous examples:

1. LM head with embedding
2. ALBERT
3. Universal Transformer

---

## 1. LM head with embedding

### Idea

A language model has two large matrices tied to the vocabulary:

| Component | Shape | Role |
|---|---|---|
| Input embedding `E` | `(vocab, d_model)` | token id → vector |
| LM head (output) `W` | `(d_model, vocab)` | vector → logits per token |

With tying, **`W = Eᵀ`**: the same matrix is used in both places.

- **Input**: row lookup, `x = E[token_id]`.
- **Output**: product with all rows, `logits = h @ Eᵀ`.

Each logit is the dot product between the final hidden state `h` and the token's embedding:

```
logit_i = h · e_i
```

The token whose embedding is most aligned with `h` gets the highest score.

### Why it works

- **Conceptual symmetry**: "what does this token mean" (input) and "which token does this vector resemble" (output) live in the same space.
- **Savings**: with a 32k vocab and `d_model` of 768, each matrix has ~24M parameters. Tying saves that amount, which can be 10-30% of a small model.
- **Better perplexity in small models** (Press & Wolf, 2017; Inan et al., 2017). Rare tokens receive gradient from both sides.

### Numeric example

Vocabulary of 3 tokens, `d_model = 2`:

```
cat  → [1.0, 0.0]
dog  → [0.9, 0.1]
car  → [0.0, 1.0]
```

With `h = [0.95, 0.05]`:

```
logit(cat) = 0.95·1.0 + 0.05·0.0 = 0.95
logit(dog) = 0.95·0.9 + 0.05·0.1 = 0.86
logit(car) = 0.95·0.0 + 0.05·1.0 = 0.05
```

"cat" wins, "dog" is close (they are similar in the space), and "car" is far.

### Code (PyTorch)

```python
import torch.nn as nn

class LM(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        # ... transformer blocks ...
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

        # weight tying: same tensor, not a copy
        self.lm_head.weight = self.tok_emb.weight
```

Verification:

```python
print(model.lm_head.weight is model.tok_emb.weight)  # True if tied
```

### Caveats

- **Initialization**: the embedding's std (commonly 0.02) now also applies to the output.
- **Scaling**: some models (T5) scale logits by `d_model^-0.5` when tying.
- **Checkpoints**: make sure tying is still active after loading the `state_dict`.
- **Large models**: the benefit shrinks and some do not tie (e.g., Llama 2 70B). Small models usually tie (GPT-2 and several SLMs; check `tie_word_embeddings` in `config.json`).

---

## 2. ALBERT

**ALBERT** (*A Lite BERT*, Lan et al., 2019) is a slimmed-down BERT built to have far fewer parameters.

### Cross-layer parameter sharing

In BERT, each of the 12 layers has its own weights. In ALBERT there is **a single Transformer block applied 12 times**:

```
BERT:    x → [Block₁] → [Block₂] → ... → [Block₁₂]   (12 sets of weights)
ALBERT:  x → [Block]  → [Block]  → ... → [Block]     (1 set, reused 12×)
```

```python
class AlbertLike(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.block = TransformerBlock(d_model, n_heads)  # just one
        self.n_layers = n_layers

    def forward(self, x):
        for _ in range(self.n_layers):
            x = self.block(x)   # same weights on every iteration
        return x
```

### Two other ALBERT ideas

**Factorized embedding**: instead of `vocab × d_model`, it uses `vocab × E` and then `E × d_model`, with a small `E` (e.g., 128).

| | Parameters (vocab 30k, d_model 768) |
|---|---|
| BERT | 30,000 × 768 ≈ 23M |
| ALBERT (E=128) | 30,000 × 128 + 128 × 768 ≈ 3.9M |

**SOP (Sentence Order Prediction)**: replaces BERT's NSP. The model must tell whether two consecutive sentences are in the right order or swapped, which forces it to learn coherence.

### Results (approximate values from the paper)

| Model | Parameters | Approx. average (GLUE/SQuAD/RACE) |
|---|---|---|
| BERT-base | ~108M | ~82.3 |
| ALBERT-base | ~12M | ~80.1 |
| BERT-large | ~334M | ~85.2 |
| ALBERT-large | ~18M | ~82.3 |
| ALBERT-xlarge | ~60M | ~85.5 |
| ALBERT-xxlarge | ~235M | ~88.7 |

- **Same width**: ALBERT loses ~2 points to BERT.
- **Same parameter budget**: ALBERT wins, since it can spend the savings on a wider model.

> The numbers are approximate. Check the original paper's table before citing them.

### Limitations

- **Fewer parameters does not mean faster**: the block runs multiple times, so FLOPs and latency are similar to a BERT of the same width. ALBERT-xxlarge is slower than BERT-large.
- **Less per-layer specialization**: early and late layers usually learn different things.
- **Rarely adopted in modern generative LLMs**, which scale parameters in decoder-only models.

---

## 3. Universal Transformer

**Universal Transformer** (Dehghani et al., 2018) also applies **the same block repeatedly**, but with a different motivation: combining the parallelism of the Transformer with the recurrent inductive bias of an RNN.

### Idea

- The same block (attention + feed-forward) is applied **T times**, as **recurrence in depth** (not in time, as in an RNN).
- At each step, all positions in the sequence are updated **in parallel**.
- It uses **position and timestep embeddings**, so the block knows which iteration it is on.

```
h⁰ → [Block + timestep 1] → h¹ → [Block + timestep 2] → h² → ... → hᵀ
          (same weights in every iteration)
```

### Adaptive Computation Time (ACT)

The big difference from ALBERT is the **adaptive number of iterations**:

- Each **position (token)** decides on its own when to stop being refined.
- "Easy" tokens stop early and "hard" tokens get more passes.
- Instead of fixed depth, depth varies with difficulty.

### Why it is interesting

- **Fewer parameters** (a single block), as in ALBERT.
- **Dynamic computation**: spends more compute where it is needed.
- **Better generalization on algorithmic and compositional tasks** (the paper reports gains on tasks such as copying, string reversal and addition, as well as translation).
- Theoretically more expressive than a fixed-depth standard Transformer (with ACT, the model is Turing-complete under certain conditions).

### Limitations

- **Computational cost** equal to stacking T distinct blocks (or higher, with ACT).
- **ACT is complex to train** and less used in practice.
- It did not become standard in LLMs, but the idea resurfaces in research on **depth-recurrent models** and adaptive computation.

---

## Comparison

| | LM head + embedding | ALBERT | Universal Transformer |
|---|---|---|---|
| **What is shared** | Vocabulary matrix | Transformer block across layers | Transformer block across steps |
| **Between which parts** | Input ↔ output | Layer 1 ↔ 2 ↔ ... ↔ N | Step 1 ↔ 2 ↔ ... ↔ T |
| **Depth** | Not affected | Fixed | Adaptive (with ACT) |
| **Reduces parameters** | Yes (~vocab × d_model) | Yes (a lot) | Yes (a lot) |
| **Reduces compute** | No | No | No |
| **Use today** | Very common | Rare in LLMs | Rare, influential idea |
| **Year** | 2016-2017 | 2019 | 2018 |

---

## When to use

- **Embedding + LM head**: almost always a good choice for small models. A nearly free gain in parameters and perplexity.
- **ALBERT / layer sharing**: only worth it if **memory** is the bottleneck. With a fixed parameter budget, smaller independent layers usually perform better.
- **Universal Transformer**: interesting for tasks requiring **iterative reasoning** or variable depth, or for research.

## Summary

Weight tying trades **parameters for regularization and savings**. In all three cases the principle is the same (one set of weights, several uses), but the savings are in **memory and parameters**, not in **compute time**.

## References

- Press & Wolf (2017). *Using the Output Embedding to Improve Language Models.*
- Inan, Khosravi & Socher (2017). *Tying Word Vectors and Word Classifiers.*
- Lan et al. (2019). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.*
- Dehghani et al. (2018). *Universal Transformers.*