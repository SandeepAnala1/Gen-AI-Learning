# How Large Language Models (LLMs) Work — Complete Technical Notes

> **Source**: *Gen AI for Data Engineers — Chapter 04: How Large Language Models (LLMs) Work* (Channel: *Ease With Data*)

---

## 1. Introduction & Core Definition

Large Language Models (LLMs) are text-based neural network models trained on massive corpora of text data with a single foundational objective: **predicting the next token (or word)**.

Capabilities such as drafting essays, composing emails, debugging code, and engaging in multi-turn dialogue stem directly from executing this next-token prediction objective at a massive scale.

### Deconstructing the Acronym "LLM"

| Element | Component | Definition & Scope |
| :--- | :--- | :--- |
| **L** | **Large** | Refers to neural networks containing **billions to trillions of parameters** (weights and biases) trained on **trillions of words** of text data. |
| **L** | **Language** | Operates on natural language (e.g., English, Hindi) or any structured sequence expressible as tokens, including programming code. |
| **M** | **Model** | A neural network architecture that takes prompts (sequences of tokens) as input and outputs the next token derived from a calculated probability distribution. |

---

## 2. Model Scaling: "How Large is Large?"

The evolution of model size and dataset volume illustrates exponential scaling over recent years:

- **GPT-1 (2018)**: 170 million parameters, trained on approximately 5 GB of book text.
- **GPT-3 (2020)**: 175 billion parameters, trained on 570 GB of text (roughly 300 billion tokens).
- **Frontier Models (Current)**: Architectures such as GPT-5, Claude, and Gemini 3 operate on parameter scales estimated in the trillions, trained across trillions of tokens.

To contextualize this scale: a basic multi-layer perceptron with 4 input nodes, 5 hidden nodes, and 3 output nodes contains roughly 43 parameters. Frontier LLMs scale this architecture to trillions of trainable parameters.

---

## 3. The Transformer Architecture

The core foundation behind modern LLMs is the **Transformer** architecture, introduced by Google in the seminal 2017 research paper, *"Attention Is All You Need"*.

### Architecture Breakdown

The Transformer architecture consists of two primary sub-components:
1. **Encoder**: Processes input text and constructs contextualized representations.
2. **Decoder**: Generates output text token-by-token in an auto-regressive fashion.

### Key Milestones (2018)
- **GPT-1 (OpenAI)**: Adapted the **Decoder-only** portion of the Transformer architecture for generative auto-regressive language modeling.
- **BERT (Google)**: Utilized the **Encoder-only** portion of the Transformer for bi-directional text understanding and contextual representations.

---

## 4. Tokenization & The Token Economy

### Vocabulary & Subword Tokenization

Each LLM operates on a fixed **vocabulary** — a predefined set of tokens it can recognize and generate.

- **Vocabulary Sizes**:
  - GPT-1: 40,000 tokens
  - GPT-3: 50,257 tokens
  - Reference: Oxford English Dictionary contains ~500,000 words.

LLMs handle vast lexicons despite vocabulary constraints via **subword tokenization**. Rather than storing entire words, words are decomposed into subword units (e.g., `"invisible"` is split into subwords `"in"` and `"visible"`).

### Tokens as "LLM Currency"

Tokens function as the core currency of LLMs:
- Text generation occurs iteratively via token-by-token processing loops.
- Computational overhead scales directly with token volume, determining financial cost in production environments.
- Tokenizers vary across model generations:
  - **GPT-5 Tokenizer vs. GPT-3 Tokenizer**: Improved efficiency in newer tokenizers compresses identical text into fewer tokens (e.g., 53 tokens in GPT-5 vs. 64 tokens in GPT-3 for the same input), reducing computational cost.
- **Rules of Thumb**:
  - In English, 1 token is approximately $3/4$ of a word ($75 \text{ words} \approx 100 \text{ tokens}$).
  - Tokens encompass complete words, subwords, punctuation marks, emojis, whitespace, and numerical digits (e.g., `123` or `789` may be single tokens, while individual digits like `0` can form distinct tokens).

---

## 5. Next-Token Prediction & Output Softmax Mechanics

When presented with an input prompt (e.g., `"The cat sat on the"`), the model generates a **probability distribution over its entire vocabulary** rather than outputting a single word directly.

```
Input Prompt ("The cat sat on the") 
  └──> Tokenizer 
        └──> Transformer Model 
              └──> Multi-Class Output Layer (Vocab Size: e.g., 50,000 Nodes)
                    └──> Softmax Activation
                          ├── "mat"   : 0.62
                          ├── "floor" : 0.18
                          ├── "chair" : 0.07
                          └── ... (Remaining Tokens)
```

### Mathematical Formulation & Softmax

The final layer of the neural network contains $V$ output nodes, where $V$ equals the vocabulary size (e.g., 50,000). The **Softmax** activation function converts raw logits into a valid probability distribution:

$$P(\text{token}_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

The sum of all probabilities across the vocabulary equals $1.0$:

$$\sum_{i=1}^{V} P(\text{token}_i) = 1.0$$

---

## 6. Sampling Parameters & Generation Control

To control how tokens are selected from the probability distribution, key sampling parameters can be tuned (typically via API or playground settings):

| Parameter | Function | Low Values | High Values |
| :--- | :--- | :--- | :--- |
| **Greedy Decoding** | Selects the single token with the highest probability. | Deterministic, repeatable output. | N/A |
| **Temperature** | Modifies output probability distribution density. | **$0.0 - 0.3$**: Highly deterministic, factual, consistent outputs (e.g., choosing `"mat"` consistently). | **$1.5 - 2.0$**: Introduces randomness and creative variation (e.g., choosing `"sofa"` or `"step"`). |
| **Top-P (Nucleus Sampling)** | Filters candidates to the smallest cumulative probability subset $P$. | **Top-P $= 0.5$**: Restricts selection strictly to tokens within the top 50% cumulative probability mass. | **Top-P $= 1.0$**: Considers the full vocabulary distribution. |

---

## 7. Prompt Personas & Conversation Roles

LLM interaction contexts maintain three standardized roles:

1. **System Role**: Defined by developers to establish model behavior, scope, guidelines, and guardrails (e.g., *"You are a specialized tax advisor chatbot"*). Hidden from end-users in standard interfaces.
2. **User Role**: Input prompts, queries, or commands provided by the human user.
3. **Assistant Role**: Responses produced by the LLM.

---

## 8. The Auto-Regressive Pipeline & Generation Loop

LLMs are **auto-regressive** ("auto" = self, "regressive" = looking back), meaning generated tokens are iteratively fed back as input for subsequent predictions.

### End-to-End Execution Pipeline

1. **Input Prompt**: Provided by the user or system.
2. **Tokenization**: Text converted into a list of subword **Tokens**.
3. **Token ID Mapping**: Tokens mapped to numerical **Token IDs**.
4. **Vector Embeddings**: Token IDs transformed into dense, high-dimensional **Vector Embeddings** via an embedding neural network to capture semantic meaning.
5. **Transformer Inference**: Embeddings processed through Transformer layers.
6. **Probability Distribution**: Multi-class output layer applies Softmax across the vocabulary.
7. **Token Selection**: Sampling strategy (Greedy, Temperature, Top-P) selects the output Token ID.
8. **Detokenization & Append**: Selected token appended to the input sequence.
9. **Loop Execution**: Process repeats until an **End of Turn (EOT)** special token is generated or the maximum context limit is reached.

---

## 9. Training Stages & Enterprise Fine-Tuning

Modern alignment and utility require three distinct training phases:

```
Raw Text Data ──> [1. Pre-Training] ──> Base Model
                                           │
Curated Datasets ──> [2. Supervised Fine-Tuning (SFT)] ──> Helpful Model
                                                               │
Human Feedback ──> [3. RLHF] ──> Production Aligned Model (Safe, Polite)
```

### The Three Training Stages

1. **Pre-Training**:
   - Trains a neural network on trillions of tokens from raw web and book corpora.
   - Resource-intensive and expensive stage.
   - Yields a **Base Model**: possesses high raw information knowledge, but lacks polite conversational skills or structured instruction-following.
2. **Supervised Fine-Tuning (SFT)**:
   - Fine-tunes the base model on high-quality, human-curated instruction-response datasets.
   - Transforms the base model into a **Helpful Model** capable of structured Q&A.
3. **Reinforcement Learning from Human Feedback (RLHF)**:
   - Uses reward models based on human feedback to reward desirable outputs and penalize harmful or incorrect ones.
   - Produces a **Safe, Polite, and Aligned Model** (e.g., ChatGPT, Claude).

### Enterprise Fine-Tuning

Enterprises take pre-trained **Base Models** or SFT checkpoints and fine-tune them on domain-specific proprietary datasets (e.g., hospital records, corporate policy documents, domain knowledge) to specialize model behavior for vertical industry applications.

---

## 10. Stateless Memory & Context Length

### Conversation Memory Architecture

LLMs possess **no inherent state or memory** across individual API calls. Multi-turn chat persistence is achieved synthetically:
- The host application resends the **entire past conversation history** (System Prompt + User Messages + Assistant Responses) alongside every new query.

### Context Length & Window Limits

- **Context Length**: The maximum number of tokens an LLM can process in a single request window (encompassing System Prompt, Chat History, and New Output Generation).
- **Context Truncation**: When conversation history exceeds the model's context limit, older turns are truncated or dropped, causing loss of earlier dialogue state.
- Modern frontier models expand context limits up to **1 million tokens** (e.g., Claude Opus), enabling analysis of entire codebases or long-form documentation.

---

## 11. Knowledge Cutoff, Hallucinations, Tools & Agents

### Knowledge Cutoff & Real-Time Data

- **Knowledge Cutoff**: Models only possess static training knowledge up to their specific training cutoff date.
- **External Tools**: Granting models access to web search, SQL databases, and calculators enables real-time access to post-cutoff information and supports **Agentic Workflows**.

### Hallucination Mechanics

- **Hallucination**: Occurs when an LLM produces fluent, authoritative, and confident statements that are factually false.
- **Cause**: LLMs prioritize statistical token fluency based on training correlations rather than strict factual verification.
- **Mitigation**: Integrating retrieval tools (RAG, web search engines, Python execution environments, database queries) grounds generation in verifiable facts.

---

## 12. Model Ecosystems & Classifications

### Open vs. Closed Models

- **Open Models**: Weights and parameters are publicly accessible and downloadable for local hosting or customization (e.g., Meta LLaMA family, DeepSeek, Mistral AI).
- **Closed Models**: Weights remain proprietary and accessible strictly via paid APIs or hosted interfaces (e.g., OpenAI GPT series, Anthropic Claude, Google Gemini).

### Reasoning & Multimodal Architectures

- **Reasoning Models**: Execute multi-step internal chain-of-thought reasoning prior to emitting response tokens, sacrificing instant output speed for elevated problem-solving capabilities in complex domains.
- **Multimodal Models**: Architectures natively capable of ingesting and producing non-text modalities (e.g., processing images, audio, and code alongside natural language text).
