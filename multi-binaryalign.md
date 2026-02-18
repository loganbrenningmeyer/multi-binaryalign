# Structured Binary Alignment with Pairwise Interaction Modeling

## Full Implementation Plan and Design Reference

---

## 0. Goal and Motivation

BinaryAlign reframes word alignment as **independent binary classification**:

$$
p(A_i) = \prod_j p(a_{ij})
$$

where each target token decision is independent.

This works well for one-to-one alignments but fails when:

* multiple target words jointly express a source word,
* alignments are **non-contiguous**,
* decisions are **interdependent** (e.g. negation, auxiliaries, multiword expressions).

Examples:

| Source    | Target          | Desired Alignment   |
| --------- | --------------- | ------------------- |
| not       | ne … pas        | both tokens         |
| give up   | abandonner      | grouped meaning     |
| would not | ne … aurait pas | structured grouping |

The core limitation is the **independence assumption**.

This extension keeps BinaryAlign’s unary formulation but adds a **pairwise interaction term**:

$$
\text{score}(S)
=
\sum_{j \in S} \phi_{ij}
+
\sum_{j<k \in S} \psi_{ijk}
$$

where:

* ( $\phi$ ): unary alignment evidence (BinaryAlign)
* ( $\psi$ ): interaction encouraging/discouraging joint selection

The alignment becomes a **set prediction problem**.

---

## 1. High-Level System Overview

For each training instance:

### Inputs

* source sentence
* target sentence
* marked source word index (i)
* gold aligned target word indices (G_i)

### Model outputs

1. Unary logits:

   ```
   φ_iℓ  → probability target token ℓ aligns to source word i
   ```

2. Pairwise logits:

   ```
   ψ_i(j,k) → compatibility of selecting target tokens j and k together
   ```

### Decoding objective

Select subset (S) maximizing:

$$
\sum_{\ell\in S} \phi_{i\ell}
+
\sum_{\ell_1<\ell_2\in S} \psi_{i,\ell_1,\ell_2}
\ 
\lambda |S|
$$

---

## 2. Data Generation and Label Construction

---

## 2.1 Existing BinaryAlign Inputs (unchanged)

Each training instance already contains:

```
{
    src_words,
    tgt_words,
    src_word_idx,
    tgt_word_idxs  # gold aligned target word indices
}
```

After tokenization:

```
input_ids        (B, L)
attention_mask   (B, L)
token_type_ids   (B, L)   # src vs tgt
word_ids         (B, L)   # token → word index or None
```

---

## 2.2 Unary Labels (φ)

Unary labels remain identical to BinaryAlign.

For each token position ℓ:

```
label = 1 if:
    token belongs to target side
    AND word_id(ℓ) ∈ aligned_target_words
else 0
```

Loss computed only over:

```
target_mask & attention_mask & (word_id != None)
```

---

## 2.3 Pairwise Labels (ψ)

Pair labels are derived from word-level alignments.

Let:

* (w(ℓ)) = target word index of token ℓ
* (G_i) = aligned target word set

Define pair label:

```
ψ_label[j,k] = 1 iff:
    w(j) ∈ G_i
    w(k) ∈ G_i
    w(j) != w(k)
```

Important rules:

### ✅ Exclude same-word pairs

Subword pairs inside the same word do not represent alignment structure.

### ✅ Only cross-word interactions matter

ψ models relationships between words, not tokenization artifacts.

---

## 2.4 Candidate Selection (Critical)

Computing ψ over all tokens is O(L²).

Instead define candidate set (C_i):

1. include all positive target tokens
2. add top-K target tokens by φ score

Typical values:

```
K = 24–32
```

Optional improvement:

* limit max subwords per word to improve diversity

---

## 3. Model Architecture

---

## 3.1 Backbone (unchanged)

Transformer encoder (e.g. mDeBERTa):

```
hidden_states: (B, L, H)
```

---

## 3.2 Unary Head (φ)

Standard BinaryAlign classifier:

```python
nn.Linear(H, 1)
```

Output:

```
phi_logits: (B, L)
```

---

## 3.3 Query Embedding (source representation)

Extract embedding representing marked source word.

Recommended:

```
q = hidden_state at <ws> token
```

Shape:

```
(B, H)
```

---

## 3.4 Candidate Target Embeddings

Gather embeddings at candidate positions:

```
t = hidden_states[:, cand_pos]
```

Shape:

```
(B, K, H)
```

---

## 3.5 Pairwise Feature Construction

For candidate tokens ($t_j$, $t_k$):

Construct symmetric features:

$$
f_{jk} =
[
q,
t_j + t_k,
t_j \odot t_k
]
$$

Where:

* sum → complementary information
* elementwise product → compatibility signal
* q → conditioning on source word

Feature dimension:

```
F = 3H
```

Tensor shapes:

```
sum_feats   (B, K, K, H)
prod_feats  (B, K, K, H)
q_expand    (B, 1, 1, H) → broadcast to (B,K,K,H)
pair_feats  (B, K, K, 3H)
```

---

## 3.6 Pairwise Head (ψ)

Simple classifier:

```python
nn.Linear(3H, 1)
```

Output:

```
psi_logits (B, K, K)
```

Constraints:

* enforce symmetry
* ignore diagonal
* ignore same-word pairs

---

## 4. Training Procedure

---

## 4.1 Total Loss

$$
\mathcal{L} =
\mathcal{L}*\phi
+
\lambda*\psi \mathcal{L}_\psi
$$

Typical:

```
λψ = 0.1 – 0.5
```

---

## 4.2 Unary Loss

Same as BinaryAlign:

```
BCEWithLogitsLoss
```

masked to target tokens.

---

## 4.3 Pairwise Loss

Problem: heavy class imbalance.

Solution:

### Negative sampling

For each example:

* include all positive pairs
* sample N negative pairs per positive (e.g. 10:1)
* skip ψ loss if fewer than 2 aligned target words

Never include:

* same-word pairs
* padding
* invalid tokens

---

## 4.4 Training Stability Tips

Recommended schedule:

1. warmup with φ-only training (optional)
2. enable ψ after unary stabilizes

Watch for:

* ψ collapsing to zero → increase λψ
* ψ overly positive → increase decoding penalty λ

---

## 5. Inference and Decoding

---

## 5.1 Candidate Selection

Same procedure as training:

* top-K by φ
* optionally include all tokens above high confidence threshold

---

## 5.2 Set Selection Objective

Score subset:

$$
\sum φ + \sum ψ - λ|S|
$$

λ discourages overly large sets.

---

## 5.3 Greedy Decoding Algorithm

Initialize:

```
S = ∅
```

Repeat:

1. compute marginal gain

$$
Δ(u) = φ_u - λ + \sum_{v∈S} ψ_{u,v}
$$

2. add token with largest positive Δ
3. stop when no positive gain remains

---

## 5.4 Word-Level Alignment Output

Convert selected tokens → words:

```
word aligned if any of its subwords selected
```

---

## 6. Evaluation Considerations

Measure:

* AER (standard)
* precision / recall
* performance on |$A_i$| > 1 cases
* non-contiguous alignment accuracy

Important analysis:

* show examples where BinaryAlign fails
* demonstrate ψ fixes these cases

---

## 7. Key Design Decisions and Tradeoffs

---

### Why keep φ at subword level?

* matches BinaryAlign training
* avoids aggregation noise
* preserves supervision density

### Why ψ across words only?

* avoids learning tokenization artifacts
* models true linguistic structure

### Why symmetric features?

* alignment interaction is unordered
* reduces parameter burden

---

## 8. Hyperparameters to Start With

```
K = 24 or 32
λψ = 0.25
negative_ratio = 10
decoding_penalty λ = 0.2–0.5
```

---

## 9. Expected Failure Modes

| Issue                        | Cause                   | Fix         |
| ---------------------------- | ----------------------- | ----------- |
| ψ always zero                | too many negatives      | increase λψ |
| overly large alignments      | ψ dominates             | increase λ  |
| missing multiword alignments | candidate set too small | increase K  |
| instability                  | early ψ training        | warmup φ    |

---

## 10. Conceptual Summary

BinaryAlign removed incorrect assumptions from earlier alignment methods:

* no softmax competition
* no span assumptions

This extension removes the remaining incorrect assumption:

> alignment decisions are independent.

The model becomes:

* unary evidence (φ)
* interaction evidence (ψ)
* structured set selection

while remaining simple, interpretable, and compatible with the BinaryAlign philosophy.

---
