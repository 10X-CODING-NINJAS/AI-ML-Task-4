# 10X Coding Ninjas - Machine Learning Track
**Student:** [NithishKumar A]
**Track:** Intermediate  
**Tasks:** Task 2 (Feature Engineering) & Task 3 (NN from Scratch)

---

## Task 2: Feature Engineering & Performance Improvement

### Objective
To demonstrate the impact of domain specific feature engineering on model predictive power by transforming raw behavioral data into high signal variables.

### Performance Summary
| Metric            | Baseline Model | Advanced Model |
| :---------------- | :------------- | :------------- |
| **Accuracy**      | 0.8082         | **0.8942**     |
| **Feature Count** | 4 (Raw)        | 28 (Engineered)|
| **Improvement**   | -              | **+8.6%**    |

### Design Choices & Insights (Explanation Depth)
The baseline model suffered from **underfitting** because it relied solely on session volume (e.g., number of product pages visited). I implemented three categories of features to resolve this:

1. **Seasonal Encoding:** One-hot encoding the `Month` allowed the model to identify high-intent periods like November (Black Friday), which raw counts ignore.
2. **Intent Intensity:** I created a ratio of `ProductRelated_Duration` to `ProductRelated` pages. This distinguishes "skimmers" from "researchers."
3. **Conversion Efficiency:** The `value_efficiency` feature (PageValue / ExitRate) acted as a "Gold Feature," providing a mathematical shortcut for the Random Forest to identify high probability buyers.

---

## Task 3: Neural Network from Scratch

### Objective
To implement a Multi Layer Perceptron (MLP) using only **NumPy** to master the underlying calculus of backpropagation and gradient descent.

### Mathematical Implementation
- **Initialization:** Xavier/Glorot Initialization was used to keep the variance of activations consistent across layers, preventing vanishing gradients.
- **Forward Pass:** Implemented linear transformations $Z = XW + b$ followed by **Sigmoid** activation.
- **Backpropagation:** Derived the **Chain Rule** to compute gradients for weights ($W$) and biases ($b$). 
  - Output Error: $dz_2 = A_2 - Y$
  - Hidden Error: $dz_1 = (dz_2 \cdot W_2^T) \times \text{sigmoid\_derivative}(A_1)$

### Comparative Analysis
| Model               |  Accuracy  |
| :------------------ | :--------- |
| **Scratch (NumPy)** | **0.7468** |
| **Sklearn (MLP)**   |   0.7338   |

**Observation:** The Scratch model achieved comparable results to `sklearn`, validating the mathematical correctness of the backpropagation logic. The slight edge in accuracy is attributed to custom-tuned learning rates for this specific dataset size.

---

## Reproducibility
To run these scripts, ensure you have the required libraries installed:
```bash
pip install -r requirements.txt
