# Neural Networks: Gen AI for Data Engineers (Video 3)
---

## 1. Why Neural Networks Matter for LLMs

- LLM names include a number such as **405B** (Llama 3.1) or **120B** (GPT OSS). This is the number of **parameters**.
- Examples mentioned:
  - GPT-3: about **175 billion** parameters (reported).
  - GPT-4 and GPT-5: not disclosed, but assumed to be in the **trillions** (GPT-5 is assumed to be around 1.6 trillion).
- **Parameters = the weights and biases of a neural network.**
- LLMs are text-based neural networks, so understanding neural networks is the foundation for LLMs and GenAI.

---

## 2. What Is an Artificial Neural Network (ANN)?

- It started from one question: *can we build software that works like the brain?*
- The brain has about **86 billion neurons**.
  - **Dendrites** receive input signals.
  - The **cell body** decides whether the combined signal is strong enough to pass on.
  - Neurons are interconnected and pass signals along.
- An artificial neuron (also called a **node** or **perceptron**) is the software version of this.

---

## 3. How a Single Artificial Neuron Works

| Component | Meaning |
|---|---|
| **Inputs (x₁, x₂, …)** | Numeric values fed into the neuron |
| **Weights (w₁, w₂, …)** | Importance of each input (higher means more important) |
| **Bias (b)** | An extra nudge added to the sum; shifts the result up or down, and ensures there is a value even if all inputs are 0 |
| **Activation function** | Applied to the sum to produce the output |
| **ŷ (y-hat)** | The neuron's output (the *predicted* value; *y* is the actual value) |

**Formula**

```
z = (w₁ · x₁) + (w₂ · x₂) + b          →   ŷ = activation(z)
```

**Steps:** multiply each input by its weight → sum them → add bias → pass through activation function → output ŷ.

### Example: Bank Loan Officer

| Input signal | Weight | Reason |
|---|---|---|
| Monthly income | **0.8** (high) | Strongest influence on approval |
| Credit score | **0.5** (medium) | Important, but less than income |
| Outstanding debts | **negative** | Debt should reduce the total |
| *Bias* | default approved loan amount | Baseline added to everyone |

The officer sums the weighted inputs plus the bias, then a function decides **loan yes or no**.

---

## 4. Activation Functions

### Purpose: add non-linearity

- **Analogy:** a fan with an on/off switch has only two states (0 or 1). Adding a **speed regulator** allows any value in between (0.3, 0.5, …).
- Without an activation function, the network can only produce linear outputs.
- With one, the network can learn **non-linear patterns** (curves, spirals).

### The two to know

| | **Sigmoid** | **ReLU** (Rectified Linear Unit) |
|---|---|---|
| Output range | 0 to 1 (S-shaped curve) | 0 for negative inputs; same as input for positive |
| Behavior | Gives a probability-like value | Blocks all negative signals |
| Typical use | **Output layer** for **binary classification** | **Hidden layers** |

Other functions such as tanh exist but are not covered in this course.

---

## 5. Fully Connected Neural Network (Layers)

```
 Input layer      Hidden layer(s)      Output layer
   (x₁) ───┬────── (h₁) ──────┐
           ├────── (h₂) ──────┼────── (ŷ)
   (x₂) ───┼────── (h₃) ──────┤
           └────── (h₄) ──────┘
```

A network is **fully connected** when every neuron connects to every neuron in the next layer.

### Input layer
- Each input neuron takes **one feature**.
- Data is fed **row by row**, not column by column.
  - Example (house price): features = area, distance from city; label = price. A row such as `(1025, 25)` goes in as two inputs.
- **No computation** happens here; it only passes the data on.

### Hidden layer(s)
- Computes: **weighted sum → add bias → activation function**.
- Called "hidden" because we don't tell it what to learn; it discovers patterns itself.
- **Early hidden layers** detect simple patterns; **deeper layers** detect complex patterns.
- More than one hidden layer makes it a **deep neural network**. LLMs are deep neural networks.

### Output layer
- Produces the prediction (ŷ). It **does perform computation** (weights, bias, activation).
- The number of output neurons depends on the task:

| Problem type | Output neurons |
|---|---|
| Binary classification (yes/no) | 1 |
| Multiclass (cat/dog/tiger) | 1 per class (3 here) |
| Regression (house price) | 1 |

---

## 6. Calculating the Number of Parameters

**Parameters = weights + biases**

- **Weights** = (neurons in layer A) × (neurons in layer B) for each pair of connected layers.
- **Biases** = one per neuron in the hidden and output layers (the input layer has none).

### Example 1: 2 inputs → 4 hidden → 1 output

- Weights: (2 × 4) + (4 × 1) = 8 + 4 = **12**
- Biases: 4 + 1 = **5**
- Total = **17**

### Example 2: 4 inputs → 5 hidden → 3 outputs

- Weights: (4 × 5) + (5 × 3) = 20 + 15 = **35**
- Biases: 5 + 3 = **8**
- Total = **43**

A tiny network already has dozens of parameters. LLMs with hundreds of billions or trillions of parameters have an enormous number of layers and neurons, which is why they can learn so much from data.

---

## 7. How a Neural Network Is Trained

Training has **three phases**, repeated in a loop:

```
Forward propagation → Loss calculation → Back propagation → (repeat)
```

1. **Forward propagation:** input goes through the network and produces a prediction.
2. **Loss calculation:** measures how far the prediction is from the actual value.
3. **Back propagation:** uses the loss to adjust weights and biases.

---

## 8. Forward Propagation: Worked Example (Spam Detection)

**Network:** 2 inputs → 2 hidden neurons (ReLU) → 1 output neuron (sigmoid). It is a binary problem: spam = 1, not spam = 0.

> Neural networks only accept **numbers**. For LLMs, text is first converted into vectors (numbers).

**Inputs:** x₁ = 0.5, x₂ = 0.8

**Hidden neuron 1** (weights 0.3 and 0.7, bias 0.1)
```
(0.5 × 0.3) + (0.8 × 0.7) + 0.1 = 0.81   →  ReLU(0.81) = 0.81
```

**Hidden neuron 2** (weights −0.2 and 0.5, bias −0.1)
```
(0.5 × −0.2) + (0.8 × 0.5) − 0.1 = 0.2   →  ReLU(0.2) = 0.2
```

**Output neuron** (weights 0.6 and 0.4, bias 0)
```
(0.81 × 0.6) + (0.2 × 0.4) + 0 = 0.566   →  Sigmoid(0.566) ≈ 0.638
```

**Result:** ŷ ≈ **0.638** (about 63.8% probability). Since 0.638 > 0.5, the email is classified as **spam**.

---

## 9. Loss Calculation

- **Loss** measures the gap between the predicted output (ŷ) and the actual output (y).
- **Goal of training: minimize the loss.**

**Example:** predicting house price from size, bedrooms, age, and distance from city, using one neuron with four weights (w₁ to w₄) and one bias.

### Loss functions

| Problem type | Loss function |
|---|---|
| **Regression** | **Mean Squared Error (MSE)** = Σ(ŷ − y)² / n |
| **Classification** | **Binary Cross Entropy** |

---

## 10. Gradient Descent

The algorithm that adjusts weights and biases to minimize loss.

### Intuition

- Picture a ball on a curve (loss on the y-axis, weight on the x-axis). We want to roll it to the lowest point.
- **Gradient = slope** at the current position.
  - Negative slope → **increase** the weight.
  - Positive slope → **decrease** the weight.
  - Slope ≈ flat (horizontal) → minimum reached.
- If you step too far, you overshoot and must come back.

**Analogy:** adjusting a shower knob with no markings. You turn it one way, then nudge it back and forth until the temperature is right.

### Update rule

```
new_weight = old_weight − (learning_rate × ∂Loss/∂weight)
```

- **Partial derivative (∂)** is used because each weight is only *partly* responsible for the loss.
- You don't compute this by hand. **Optimizer functions** in libraries such as **PyTorch** do it, which matters when there are billions of parameters.

### Learning rate
- The **step size** of each adjustment.
  - High: big, fast steps (risk of overshooting).
  - Low: small, slow steps.

---

## 11. Types of Gradient Descent

| Type | How it works | Best for |
|---|---|---|
| **Batch** | Feed **all rows**, compute loss, then adjust weights once | Smaller datasets |
| **Stochastic (SGD)** | Feed **one row**, compute loss, adjust weights, then move to the next row | Larger datasets |
| **Mini-batch** | Split data into small batches (e.g. 3 rows each); compute loss and adjust after each batch | Common compromise |

### Epoch
- **One epoch = one full pass through the entire dataset.**
- Training repeats over many epochs until the loss stops improving.
- Example (SGD): 100 epochs × 10 records = 1,000 weight adjustments.

---

## 12. Back Propagation

- Happens **after the loss is calculated**: weights and biases are adjusted starting from the output and moving backward through **all layers**.
- In deep networks this is complex, so optimizers and libraries (like PyTorch) handle it.

### Training vs. testing

| Phase | What happens |
|---|---|
| **Training** | Forward propagation → loss → back propagation, looped over many epochs |
| **Testing** | **Forward propagation only**; no weight adjustment |

---

## 13. Trainable Parameters vs. Hyperparameters

| | **Trainable parameters** | **Hyperparameters** |
|---|---|---|
| What | Weights and biases | Settings *you* choose |
| Learned from data? | Yes | No |
| Examples | w₁, w₂, …, b | Learning rate, number of epochs, batch size, number of hidden layers, neurons per layer, choice of optimizer |

The "405B" in an LLM's name counts **trainable parameters**.

---

## 14. Training Loss Curve

- Plot **loss (y-axis)** against **training steps/epochs (x-axis)**.
- A healthy curve **decreases steadily** and then flattens, which indicates correct training and good hyperparameters.

```
Loss
 │\
 │ \
 │  \___
 │      \_____________
 └─────────────────────── Epochs
```

---

## 15. Scope Notes

- Everything above applies to a standard ANN for **non-spatial data**.
- **Images (spatial data)** use **CNNs** (Convolutional Neural Networks). They are not covered in this course.
- For text/LLMs, upcoming topics are **RNN** and **LSTM**.
- **Next video:** basics of LLMs.

---

## Quick Revision Cheatsheet

- **Parameters** = weights + biases.
- **Neuron:** `ŷ = activation(Σ wᵢxᵢ + b)`
- **Sigmoid** → output layer (binary); **ReLU** → hidden layers.
- **Input layer** does no computation.
- **Deep network** = more than one hidden layer.
- **Training loop:** forward → loss → backward → repeat for many epochs.
- **MSE** for regression; **Binary Cross Entropy** for classification.
- **Gradient descent:** `w_new = w − lr × gradient`
- **Batch / SGD / Mini-batch** differ in *when* weights are updated.
- **Hyperparameters** are set by you; **parameters** are learned from data.
