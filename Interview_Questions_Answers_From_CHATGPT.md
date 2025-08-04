### 🔹 Question 1

Absolutely! Here's a simple **PyTorch example** demonstrating how to use `softmax` for a classification task.

---

### 🔹 Goal

Let’s say we have a model that outputs **logits** (unnormalized scores), and we want to:

1. Convert them to probabilities using `softmax`
2. Calculate loss using `cross_entropy`
3. Backpropagate and update the model

---

### 🔹 Code Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dummy input (e.g. 4 features)
x = torch.tensor([[0.5, 1.0, -0.5, 2.0]], requires_grad=True)

# Dummy linear classifier (4 inputs -> 3 classes)
model = nn.Linear(4, 3)

# Forward pass: logits (unnormalized scores)
logits = model(x)  # shape: (1, 3)

# Apply softmax to get probabilities
probs = F.softmax(logits, dim=1)

print("Logits:", logits)
print("Probabilities (softmax):", probs)

# Let's assume the true class is class 2 (index 1-based, so label = 1)
true_label = torch.tensor([1])

# Cross-entropy loss (automatically applies log-softmax internally)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, true_label)

# Backpropagate
loss.backward()

print("Loss:", loss.item())
print("Gradients on input x:", x.grad)
```

---

### 🔹 Notes

* `nn.CrossEntropyLoss()` combines `log_softmax` and `nll_loss`, so **you don't apply `softmax` before feeding logits to it**.
* You can use `F.softmax(logits, dim=1)` **only for viewing** probabilities, not for loss computation.
* `x.requires_grad=True` to demonstrate gradient computation (usually, the model's parameters are updated, not the input).

---

### 🔹 Output (example)

```text
Logits: tensor([[ 0.3,  1.2, -0.7]], grad_fn=<AddmmBackward0>)
Probabilities (softmax): tensor([[0.269, 0.655, 0.076]], grad_fn=<SoftmaxBackward0>)
Loss: 0.423
Gradients on input x: tensor([...])
```

Let me know if you want this extended into a full training loop or with a real dataset (like MNIST)!


### 🔹 Question 2

That's an **excellent observation**, and it's a very common question!

You're right to notice that in the simplified cross-entropy loss:

$$
\mathcal{L} = -\log(\hat{y}_k)
$$

it **only uses the predicted probability of the true class** $\hat{y}_k$. So naturally, it might seem like the other classes don’t affect the loss or gradients.

---

### 🔹 But Here’s the Key Point:

Even though **only the true class's log probability is used in the final loss value**, the **gradient of that loss** (after backpropagation through softmax) **does depend on all logits**.

---

### 🔹 Why? Because of Softmax!

Let’s recall that the predicted probabilities $\hat{y}_i$ are:

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}
$$

where $z_i$ are the **logits** (raw model outputs).

This means:

* Changing **any one logit** $z_j$ will change **all** $\hat{y}_i$
* So the gradient of $\mathcal{L}$ with respect to any $z_j$ (not just $z_k$) is **non-zero**

---

### 🔹 Gradient Formula (w\.r.t. logits)

Let $z_i$ be the logit for class $i$, and the true class be $k$. Then:

$$
\frac{\partial \mathcal{L}}{\partial z_i} =
\begin{cases}
\hat{y}_i - 1 & \text{if } i = k \\
\hat{y}_i     & \text{if } i \neq k
\end{cases}
$$

This means:

* For the **correct class**, gradient = $\hat{y}_k - 1$ → update to increase confidence
* For **incorrect classes**, gradient = $\hat{y}_i$ → update to decrease their probability

So **all classes contribute to weight updates**, not just the correct one.

---

### 🔹 Summary

| Misunderstanding                                           | Reality |
| ---------------------------------------------------------- | ------- |
| Only true class contributes to updates                     | ❌ False |
| Other classes also influence the loss gradient via softmax | ✅ True  |

---

Let me know if you'd like to **visualize this with PyTorch code** — it's easy to confirm this by printing gradients of the logits!
