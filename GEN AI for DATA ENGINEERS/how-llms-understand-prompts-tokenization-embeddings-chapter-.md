# Comprehensive Engineering Notes: How LLMs Understand Prompts (Tokenization & Embeddings)

As presented in Chapter 05 of the *Gen AI for Data Engineers* video series by *Ease With Data*, Large Language Models (LLMs) and underlying Transformer architectures are fundamentally mathematical engines. Neural networks process numerical inputs through matrix multiplications, calculus, and linear algebra operations; they possess no native capability to interpret raw human text, individual characters (such as `'a'`), or full string words (such as `'cat'`). 

This document provides complete, structured technical notes covering the end-to-end mechanism of converting raw text prompts into high-dimensional numerical vectors while strictly preserving semantic meaning and sequential structure. This pipeline forms the mandatory input layer for downstream Transformer processing and Retrieval-Augmented Generation (RAG) architectures.

---

## 1. Executive Technical Overview: Converting Text to Numerical Representation

The core objective of tokenization and embedding generation is to transform unformatted text into a structured numeric format that captures both word meaning and sequential position. 

### The Complete End-to-End Processing Pipeline

The transformation from raw prompt text to Transformer input follows a strict, sequential four-stage pipeline:

1. **Text Segmentation & Tokenization**: Raw input text (for example, the 7-token sentence `"the cat sat on the mat ."`) is segmented into discrete units called tokens (words, subwords, or characters) based on a pre-trained vocabulary. Each unique token is assigned an integer index called a **Token ID**.
2. **Dense Vector Embedding Lookup**: Token IDs act as lookup indices into a pre-trained **Embedding Matrix** (weights matrix), retrieving a fixed-size dense numerical vector that encodes the semantic concept of each token.
3. **Positional Encoding Integration**: To preserve sequential context, a **Positional Embedding** vector of identical dimension is mathematically added to each token embedding vector.
4. **Transformer Layer Processing**: The combined semantic-positional dense vectors are passed directly into the multi-head attention mechanisms and feed-forward layers of the Transformer.

The table below summarizes each stage of the pipeline using the sample sequence `"the cat sat on the mat ."`.

| Pipeline Stage | Input Artifact | Processing Mechanism | Output Artifact |
| :--- | :--- | :--- | :--- |
| **Stage 1: Tokenization** | Raw text string: `"the cat sat on the mat ."` | Vocabulary lookup against pre-trained token dictionary | 7 Tokens with unique Token IDs (e.g., `"cat"` $\rightarrow$ ID `9246`) |
| **Stage 2: Embedding Lookup** | Token ID `9246` | Dense matrix lookup (`embedding_matrix[9246]`) | 768-Dimensional Semantic Dense Vector |
| **Stage 3: Positional Encoding** | Semantic Dense Vector + Index Position | Element-wise vector addition ($\vec{E}_{\text{token}} + \vec{E}_{\text{position}}$) | Position-Aware Integrated Embedding Vector |
| **Stage 4: Model Execution** | Position-Aware Vector Sequence | Parallel matrix multiplication in Multi-Head Self-Attention | Contextualized Hidden States |

---

## 2. Linear Algebra Fundamentals and Geometric Vector Spaces

To understand how LLMs represent semantic meaning numerically, it is essential to ground the architecture in linear algebra and coordinate geometry.

### Definition of Vectors and Dimensions

A vector is defined as an ordered list of numbers representing a point in a multi-dimensional space:

* **2-Dimensional Vector ($2\text{D}$)**: Expressed as an ordered list of two numbers, such as $\vec{v} = (3, 2)$. In Cartesian coordinate geometry, this represents a point on a 2D plane where $x = 3$ and $y = 2$, visualised as a directed arrow drawn from the origin $(0, 0)$ to $(3, 2)$.
* **3-Dimensional Vector ($3\text{D}$)**: Expressed as an ordered list of three numbers, such as $\vec{v} = (1, 2, 5)$, representing spatial coordinates across the $X$, $Y$, and $Z$ axes.
* **High-Dimensional Vector ($d\text{-D}$)**: Expressed as an ordered list of $d$ numbers, such as a 768-dimensional vector $(v_1, v_2, v_3, \dots, v_{768})$. While humans cannot visually render spaces beyond three dimensions, digital computers handle $768$-dimensional vectors using the exact same linear algebra operations applicable to $2\text{D}$ and $3\text{D}$ vectors.

### Vector Operations in Neural Networks

Neural networks rely primarily on three linear algebraic operations across vector representations:

1. **Vector Addition**: Performed element-wise across corresponding dimensions, as demonstrated in the lecture:
   $$\begin{pmatrix} 3 \\ 2 \end{pmatrix} + \begin{pmatrix} -1 \\ 2 \end{pmatrix} = \begin{pmatrix} 3 + (-1) \\ 2 + 2 \end{pmatrix} = \begin{pmatrix} 2 \\ 4 \end{pmatrix}$$
2. **Vector Subtraction**: Performed element-wise to compute spatial differences or offsets between points in vector space.
3. **Directional Comparison**: Evaluating spatial orientation and angular alignment between vectors using trigonometric functions such as cosine similarity.

### Geometric Encoding of Meaning: "Numbers Become Geometry"

In an embedding space, numerical values correspond directly to spatial locations. As demonstrated in the lecture transcript, tokens with similar meanings yield dense vectors with nearly identical numerical components, causing them to point in almost identical directions in vector space.

The following comparison illustrates how semantic proximity translates into numerical similarity across vector dimensions:

| Vector Dimensions | Token Vector: `"cat"` | Token Vector: `"dog"` | Token Vector: `"the"` |
| :--- | :--- | :--- | :--- |
| **Dimension 1** | $0.91$ | $0.89$ | $-0.42$ |
| **Dimension 2** | $-0.13$ | $-0.15$ | $0.88$ |
| **Dimension 3** | $0.55$ | $0.52$ | $-0.04$ |
| **Dimension $d$** | $-0.52$ | $-0.50$ | $0.12$ |

Because the vectors for `"cat"` and `"dog"` share closely aligned values across corresponding dimensions ($0.91$ vs $0.89$, $-0.13$ vs $-0.15$, $0.55$ vs $0.52$), their geometric distance is small and their directional alignment is high. Conversely, structural tokens like `"the"` produce vastly different numerical values, pointing toward an entirely different region of the vector space.

---

## 3. Semantic Vector Dimensions and Hyperparameter Scaling

To conceptualise how dense vectors encode specific concepts, consider a simplified 4-dimensional vector space where each dimension represents a distinct semantic trait scaled between $-1.0$ and $+1.0$:

* **Dimension 1**: Size / Scale (`Big`)
* **Dimension 2**: Threat Level (`Dangerous`)
* **Dimension 3**: Surface Texture (`Furry`)
* **Dimension 4**: Domestication (`Pet`)

### Conceptual 4-Dimensional Feature Matrix

The table below maps four animal tokens across these feature dimensions to illustrate multi-dimensional semantic mapping:

| Token | Dimension 1: `Big` | Dimension 2: `Dangerous` | Dimension 3: `Furry` | Dimension 4: `Pet` | Synthesized Semantic Profile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `"cat"` | $-0.7$ | $-0.6$ | $+0.8$ | $+0.9$ | Small, harmless, furry, domestic pet |
| `"kitten"` | $-0.9$ | $-0.7$ | $+0.9$ | $+0.9$ | Very small, harmless, furry, domestic pet |
| `"tiger"` | $+0.8$ | $+0.9$ | $+0.7$ | $-0.9$ | Large, highly dangerous, furry, wild animal |
| `"shark"` | $+0.7$ | $+0.9$ | $-0.9$ | $-0.9$ | Large, highly dangerous, non-furry, wild aquatic animal |

### Analysis of Semantic Proximity

1. **`"cat"` vs. `"kitten"`**: Both vectors exhibit nearly identical numerical signatures across all four dimensions ($[-0.7, -0.6, +0.8, +0.9]$ vs $[-0.9, -0.7, +0.9, +0.9]$), placing them in the same localized cluster in vector space.
2. **`"tiger"` vs. `"shark"`**: Despite belonging to different species classes (mammal vs. fish), both tokens share high positive values on `Big` and `Dangerous` and negative values on `Pet`, aligning them along danger and size dimensions in vector space.

### Dimensionality as a Pre-Training Hyperparameter

While a 4-dimensional vector space is sparse and captures limited conceptual nuances, real-world LLMs expand feature dimensions to capture rich semantic subtleties:

* **GPT-2 Architecture**: Employs $768$-dimensional embedding vectors per token ($d_{\text{model}} = 768$).
* **Dimensional Scaling Trade-Off**: Increasing dimensions allows a model to capture finer semantic nuances. However, hyperparameter size cannot be expanded arbitrarily (for example, to $1,000,000$ dimensions) because higher dimensionality significantly increases neural network training complexity, memory footprint, and computational overhead. Dimensionality is a fixed hyperparameter established prior to model pre-training.

---

## 4. Mathematical Vector Similarity: Cosine Similarity Mechanics

To quantitatively measure semantic similarity between two token embedding vectors $\vec{A}$ and $\vec{B}$, models use **Cosine Similarity**. 

### Trigonometric Derivation and Formula

Cosine similarity calculates the cosine of the angle $\theta$ between two vectors in a multi-dimensional space:

$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{d} A_i B_i}{\sqrt{\sum_{i=1}^{d} A_i^2} \sqrt{\sum_{i=1}^{d} B_i^2}}$$

### Spatial Interpretation of Angular Values

The relationship between spatial angle and cosine similarity is structured as follows:

* **$\cos(0^\circ) = 1.0$**: The angle between the two vectors is $0^\circ$. They point in the exact same direction, indicating maximum semantic similarity.
* **$\cos(90^\circ) = 0.0$**: The vectors are orthogonal (perpendicular). There is no linear correlation or shared semantic meaning between the tokens.
* **$\cos(180^\circ) = -1.0$**: The vectors point in diametrically opposite directions, indicating complete semantic opposition.

### Quantitative Comparison Examples

As highlighted in the video transcript, evaluating token vectors through cosine similarity yields clear numerical boundaries:

* $\text{Cosine Similarity}(\text{Vector}_{\text{cat}}, \text{Vector}_{\text{kitten}}) \approx +0.8$ (High directional alignment $\rightarrow$ Strong semantic similarity).
* $\text{Cosine Similarity}(\text{Vector}_{\text{cat}}, \text{Vector}_{\text{shark}}) \approx -0.8$ (Divergent directional alignment $\rightarrow$ Weak/opposite semantic correlation).

---

## 5. Tokenization Methodologies and Architecture Trade-offs

Tokenization is the process of decomposing raw text into discrete tokens before numerical mapping. As explained in the tutorial, three primary tokenization paradigms exist, each presenting distinct engineering trade-offs.

### Comparative Evaluation of Tokenization Paradigms

The table below compares character, word, and subword tokenization methods:

| Tokenization Paradigm | Granularity Unit | Example Segmentation (`"invisible"`) | Key Advantages | Major Demerits & Engineering Bottlenecks |
| :--- | :--- | :--- | :--- | :--- |
| **Character-Level** | Individual characters | `['i', 'n', 'v', 'i', 's', 'i', 'b', 'l', 'e']` | Compact vocabulary size; eliminates out-of-vocabulary (OOV) errors. | **Massive token inflation**. A 1,000-word text generates thousands of tokens, dramatically increasing LLM compute cycles, processing latency, and financial costs. |
| **Word-Level** | Full words split by whitespace | `['invisible']` | Intuitive mapping; preserves full word units. | 1. **Vocabulary Explosion**: Morphological variants (e.g., `play`, `playing`, `plays`) require separate entries, leading to unmanageable vocabulary sizes.<br>2. **Whitespace Dependency**: Fails on non-delimited languages like Chinese. |
| **Subword-Level (BPE/BBPE)** | Frequent subword chunks/morphemes | `['in', 'visible']` | **Optimal balance**. Handles rare/unseen words via sub-units while keeping token counts low. | Requires sophisticated subword segmentation algorithms during pre-training. |

### Subword Tokenization & Byte Pair Encoding (BPE)

Subword tokenization addresses the limitations of character-level and word-level approaches by splitting words into frequent subword units. 

For instance, the word `"indivisible"` is tokenized by OpenAI's GPT-2 tokenizer into two discrete subword tokens: `["indivi", "isible"]`. During generation, the model processes these individual subwords and subsequently re-combines them into the unified term `"indivisible"`. Modern architectures predominantly utilize **Byte Pair Encoding (BPE)** or **Byte-Level Byte Pair Encoding (BBPE)**.

### Cross-Lingual Tokenization Efficiency Disparities

A critical finding highlighted in the video is that tokenization efficiency varies significantly across languages due to the training distribution of the tokenizer vocabulary.

When encoding the identical semantic sentence (*"Hello my name is Alice i love learning."*) across different languages using OpenAI's `tiktoken` library:

* **English Encoding**: Consumes **11 tokens**.
* **Arabic Encoding**: Consumes **23 tokens**.

Because the underlying vocabulary is predominantly trained on English text, non-English languages (such as Arabic or Chinese) are fragmented into smaller, more granular subword chunks. Consequently, processing non-English prompts requires substantially more tokens for the exact same semantic payload, resulting in higher latency, elevated API costs, and faster consumption of the context window.

---

## 6. Practical Implementation: Tokenization with OpenAI Tiktoken

To demonstrate subword tokenization programmatically, the tutorial utilizes OpenAI's high-performance `tiktoken` Python library.

### Library Installation and Initialization

```python
# Step 1: Install tiktoken library in Google Colab environment
!pip install tiktoken

# Step 2: Import encoding retriever and initialize cl100k_base vocabulary
import tiktoken

# Load cl100k_base encoding (used by GPT-4 and OpenAI embedding models, vocabulary ~100k)
enc = tiktoken.get_encoding("cl100k_base")

# Inspect vocabulary capacity
print("Max Token Index Value:", enc.max_token_value)
# Output: 100256 (representing the total token capacity of the vocabulary)
```

### Text Encoding and Subword Decomposition Walkthrough

When tokenizing a string containing a proper name not present as a single word in the pre-trained vocabulary (e.g., `"Shoubam"`), the tokenizer breaks the string into subword units:

```python
# Define input text containing an out-of-vocabulary name
input_text = "my name is Shoubam"

# Encode text string into a list of integer Token IDs
token_ids = enc.encode(input_text)
print("Generated Token IDs:", token_ids)
# Output: List of discrete integer token IDs corresponding to each subword token

# Inspect individual subword token byte representations
# Inspecting token ID 3804 (at index 3) and token ID 5721 (at index 4)
subword_part1 = enc.decode_single_token_bytes(token_ids[3])
subword_part2 = enc.decode_single_token_bytes(token_ids[4])

print("Token ID 3804 decodes to:", subword_part1)  # Output: b'su'
print("Token ID 5721 decodes to:", subword_part2)  # Output: b'ham'

# Decode full sequence back to original text
decoded_text = enc.decode(token_ids)
print("Decoded String:", decoded_text)
# Output: "my name is Shoubam"
```

---

## 7. Practical Implementation: Dense Embedding Matrix Lookup with PyTorch & Transformers

Once text is tokenized into Token IDs, the next step is looking up those IDs in the model's pre-trained embedding matrix to retrieve dense feature vectors.

### Model and Tokenizer Synchronization Requirement

A fundamental rule established in the lecture is that **the tokenizer and the embedding matrix must belong to the exact same model architecture and vocabulary state**. 

One cannot pair a GPT-3 embedding matrix with a GPT-2 tokenizer. Because Token IDs act as direct row indices into the embedding matrix lookup table, mismatching the vocabulary alters the row assignments, yielding invalid vector lookups.

### Token-to-Embedding Extraction Walkthrough

```python
# Step 1: Install Hugging Face Transformers library
!pip install transformers

# Step 2: Import tiktoken and GPT2Model from transformers
import tiktoken
import torch
from transformers import GPT2Model

# Step 3: Initialize matching GPT-2 tokenizer encoding and GPT-2 pre-trained model
enc = tiktoken.get_encoding("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Step 4: Tokenize target word "cat" to retrieve its Token ID
target_word = "cat"
token_ids = enc.encode(target_word)
print("Token ID for 'cat':", token_ids)
# Output: [9246]

# Step 5: Access the model's pre-trained word token embedding matrix (wte weights)
embedding_matrix = model.wte.weight

# Step 6: Perform matrix lookup using Token ID 9246
cat_token_id = token_ids[0]
cat_vector = embedding_matrix[cat_token_id]

# Step 7: Verify tensor dimensions
print("Cat Embedding Tensor Shape:", cat_vector.shape)
# Output: torch.Size([768])
```

The output confirms that the token ID `9246` retrieves a PyTorch tensor of shape `torch.Size([768])`, matching GPT-2's fixed 768-dimensional model architecture.

---

## 8. Positional Embeddings and Sequence Preservation

Semantic vector lookups alone are insufficient for full language comprehension because token embeddings do not natively encode word order.

### The Problem of Word Order Sensitivity

Consider the following two sentences containing identical tokens:

1. `"the dog bit the man"`
2. `"the man bit the dog"`

If a model only considers token embedding vectors, both sentences produce identical unordered sets of vectors ($\{\text{vector}_{\text{the}}, \text{vector}_{\text{dog}}, \text{vector}_{\text{bit}}, \text{vector}_{\text{the}}, \text{vector}_{\text{man}}\}$). However, the underlying semantic meanings are completely inverted due to the swapped positions of `"dog"` and `"man"`.

### Positional Embedding Vector Addition

To inject spatial position awareness into the sequence, the architecture computes a **Positional Embedding** vector for each index position in the sequence and adds it directly to the semantic token embedding vector:

$$\vec{E}_{\text{final}} = \vec{E}_{\text{token}} + \vec{E}_{\text{position}}$$

The element-wise vector addition combines the semantic meaning vector ($\vec{E}_{\text{token}}$) with the token's position in the sequence ($\vec{E}_{\text{position}}$), producing a final integrated embedding vector that is fed directly into the Transformer's self-attention layers.

### Complete Pipeline Summary

1. **Input Prompt**: Raw text string.
2. **Tokenization**: Text segmented into tokens; unique Token IDs assigned from pre-trained vocabulary.
3. **Semantic Lookup**: Token IDs used to look up dense semantic vectors ($\text{Dimension} = d$) in the Embedding Matrix.
4. **Positional Encoding**: Positional embedding vectors added element-wise to capture sequence order.
5. **Transformer Execution**: Combined final embeddings passed into Transformer self-attention mechanisms for downstream processing.
