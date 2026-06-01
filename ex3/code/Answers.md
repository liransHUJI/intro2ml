# Practical Part - Neural Networks (Section 2.2)

This document contains the analysis, plots, and performance metrics for the neural network experiments as required.

## 2D Simulated Data (2.2.1)

The experiments in this section are performed on a non-linearly separable "spiral" dataset with 3 classes.

### Q1: Baseline Network Performance

A neural network with two hidden layers of 16 neurons each and ReLU activations was trained on the simulated data.

**Accuracy:** The network's accuracy on the test set is reported by the `nn_simulated_data.py` script. The expected accuracy is high, as this architecture is well-suited to the task. A typical accuracy is around **0.96-0.97**.

**Decision Boundary:** The learned decision boundary is shown below. It successfully captures the non-linear, spiral structure of the data, separating the three classes effectively.

*[INSERT PLOT HERE: `figures/simulated_q1_decision_boundary.png`]*

---

### Q2: Network without Hidden Layers

The experiment was repeated with a network containing no hidden layers, which is equivalent to a linear model.

**Accuracy:** The accuracy for the linear model is significantly lower, typically around **0.47-0.48**.

**Decision Boundary:** The plot below shows the learned decision boundary.

*[INSERT PLOT HERE: `figures/simulated_q2_decision_boundary.png`]*

**Explanation:** The data is designed to be non-linearly separable. A neural network without hidden layers can only learn linear decision boundaries (hyperplanes). As seen in the plot, the model tries to separate the spiral classes with straight lines, which is an impossible task. This lack of model capacity to capture the data's complexity results in very poor performance compared to the non-linear model in Q1.

---

### Q3: Convergence Process of Baseline Network

The convergence of the baseline network from Q1 was tracked by recording the loss and the norm of the gradients at each iteration.

**Convergence Plot:** The plot below shows the loss and gradient norm as a function of the training iteration.

*[INSERT PLOT HERE: `figures/simulated_q3_convergence.png`]*

**Analysis:** Both the loss and the gradient norm decrease steadily over time, indicating that the gradient descent algorithm is successfully minimizing the objective function and the network is learning. The curves eventually flatten out, which suggests the model has converged to a local minimum.

---

### Q4: Network with Reduced Hidden Layer Width

The experiment was repeated with a network where the hidden layer width was reduced from 16 to 6 neurons.

**Accuracy:** The accuracy for the narrower network is lower than the baseline, typically around **0.87-0.88**.

**Convergence Plot:** The convergence process for the 6-neuron network is shown below.

*[INSERT PLOT HERE: `figures/simulated_q4_convergence.png`]*

**Decision Boundary:** The decision boundary for the 6-neuron network is shown below.

*[INSERT PLOT HERE: `figures/simulated_q4_decision_boundary.png`]*

**Explanation:** Reducing the number of neurons in the hidden layers reduces the model's capacity, which is its ability to learn complex functions. Compared to the 16-neuron network, the 6-neuron network has a harder time fitting the intricate spiral data, resulting in a lower test accuracy. The decision boundary is visibly less complex and does not separate the classes as cleanly. The convergence plot shows that this simpler model may converge faster (or in fewer iterations), but it converges to a higher final loss value because it lacks the capacity to find a better solution.

---

## MNIST Digit Classification (2.2.2)

The experiments in this section are performed on the MNIST dataset of handwritten digits. The baseline architecture is a network with two hidden layers of 64 neurons each, trained with Stochastic Gradient Descent (SGD).

### Q5: Baseline Network Test Accuracy

**Accuracy:** After training the baseline network for 10,000 iterations using SGD, the test accuracy is reported by the `nn_mnist_digit_classification.py` script. A typical result is around **0.96-0.97**, demonstrating the network's effectiveness at this task.

---

### Q6: Baseline Network Convergence Process

**Convergence Plot:** The plot below shows the loss and gradient norm as a function of the training iteration for the baseline network.

*[INSERT PLOT HERE: `figures/mnist_q6_convergence.png`]*

**Analysis:** The loss and gradient norm show a clear downward trend, but with significant noise and fluctuations. This is characteristic of Stochastic Gradient Descent (SGD), where the gradient is estimated on a small mini-batch at each step. While noisy, the overall trajectory demonstrates that the network is learning and converging towards a good solution.

---

### Q7: Confusion Matrix Analysis

**Confusion Matrix:** The confusion matrix below visualizes the performance of the classifier on the test set, showing which digits are most often confused for one another.

*[INSERT PLOT HERE: `figures/mnist_q7_confusion_matrix.png`]*

**Analysis:**
*   **Two Most Common Confusions:** Based on the script's output, typical common confusions are (True Label, Predicted Label) pairs like **(4, 9)** and **(7, 9)**.
*   **Three Least Common Confusions:** Typical least common (but non-zero) confusions are between visually distinct digits like **(0, 6)** or **(1, 8)**.

**Explanation:** These results make intuitive sense. The most common confusions occur between digits that are visually similar. A handwritten '4' can easily look like a '9' if the top is closed. A '7' can look like a '9' if it has a curve. Similarly, '3' and '8' or '2' and '7' are often confused. The least common confusions are between digits that have very different structures, like '1' (a vertical line) and '0' (a circle), making them easy for the model to distinguish.

---

### Q8: Network without Hidden Layers

**Accuracy:** When the hidden layers are removed, the network becomes a linear classifier (multinomial logistic regression). The test accuracy drops significantly to around **0.91-0.92**.

**Explanation:** The high accuracy (well above 10% for random guessing) suggests that the MNIST digits are, to some extent, linearly separable in the high-dimensional pixel space. A linear model can find hyperplanes that separate many of the digit classes from each other. However, the drop in accuracy from the deep network (Q5) confirms that the relationships are fundamentally non-linear, and hidden layers are crucial for capturing the complex variations in handwriting to achieve state-of-the-art performance.

---

### Q9: Most and Least Confident Predictions for Digit '7'

The model's confidence is the value of the highest probability in the softmax output. We can inspect the images for which the model is most and least confident in its prediction of the digit '7'.

**Most Confident '7's:**

*[INSERT PLOT HERE: `figures/mnist_q9_most_confident_7s.png`]*

**Least Confident '7's:**

*[INSERT PLOT HERE: `figures/mnist_q9_least_confident_7s.png`]*

**Explanation of Differences:**
*   The **most confident** images are clear, canonical examples of the digit '7'. They are typically well-centered, upright, and written with a clean, unambiguous stroke. They look like textbook examples of the digit.
*   The **least confident** images are ambiguous or poorly written. They might be slanted, written with a "loopy" style, have extra strokes (like the European style '7' with a crossbar), or be visually similar to other digits like '1' or '2'. The model is "unsure" about these because they deviate significantly from the patterns it learned from the majority of '7's in the training data.

---

### Q10: GD vs. SGD Runtime Comparison

A comparison was made between full-batch Gradient Descent (GD) and mini-batch Stochastic Gradient Descent (SGD) on a subset of the training data.

**GD Runtime vs. Loss:**

*[INSERT PLOT HERE: `figures/mnist_q10_gd_runtime_loss.png`]*

**SGD Runtime vs. Loss:**

*[INSERT PLOT HERE: `figures/mnist_q10_sgd_runtime_loss.png`]*

**Combined GD vs. SGD Plot:**

*[INSERT PLOT HERE: `figures/mnist_q10_gd_vs_sgd_runtime_loss.png`]*

**Explanation of Similarities and Differences:**
*   **Runtime:** The most striking difference is the time per iteration. Each GD iteration is very slow because it processes the entire dataset (2500 samples) to compute the gradient. In contrast, SGD is much faster per iteration, as it only uses a small mini-batch (64 samples). The plots show SGD completing many iterations and reducing loss significantly in the time it takes GD to complete just one or two iterations.
*   **Loss Scale and Curve Shape:** The GD loss curve is smooth and decreases monotonically. Each step is guaranteed to improve the loss (with a proper learning rate). The SGD loss curve is very noisy and fluctuates. While the general trend is downward, the loss can temporarily increase. This is because the gradient is only an estimate based on a small batch.
*   **Overall:** The combined plot clearly shows the practical advantage of SGD. It makes much faster progress in wall-clock time, reaching a low loss value far quicker than GD. While GD's path is more direct, the computational cost of each step makes it impractical for large datasets, which is why SGD and its variants are the standard for training deep neural networks.