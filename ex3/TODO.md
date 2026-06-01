# Exercise 3 TODO: Gradient-Based Learning and Neural Networks

Source PDF: `ex3 - GD and NN.pdf`

## Full-Grade Requirements Checklist

Use this section as the final grading checklist. Everything here must be complete for a full-credit submission.

### A. Required Submission Package

- [ ] Create a single zip named `ex3_ID.zip`, replacing `ID` with your ID.
- [ ] Put all required files directly in the zip root, without directories.
- [ ] Include exactly one `Answers.pdf` with all written answers and required plots.
- [ ] Include the required Python files:
  - `gradient_descent.py`
  - `gradient_descent_investigation.py`
  - `learning_rate.py`
  - `modules.py`
  - `neural_network.py`
  - `nn_loss_functions.py`
  - `nn_mnist_digit_classification.py`
  - `nn_simulated_data.py`
  - `nn_utils.py`
  - `nn_modules.py`
  - `stochastic_gradient_descent.py`
- [ ] Do not rely on separate plot files for grading.
  - The PDF explicitly says plots included as separate files are considered not provided.
  - Generated `figures/*.html` and `figures/*.png` are local working artifacts only.
- [ ] Submit the zip to the designated Moodle activity before the deadline shown there.

### B. Required Theoretical Answers in Answers.pdf

- [ ] Q1: Prove that a nonnegative weighted sum of convex functions is convex.
- [ ] Q2: Give a counterexample showing that composition of two convex functions need not be convex.
- [ ] Q3: Prove hinge loss is convex in `(w, b)`.
- [ ] Q4: Derive a valid sub-gradient of hinge loss.
- [ ] Q5: Prove the sum of sub-gradients is a sub-gradient of the sum.
- [ ] Q6: Derive a sub-gradient for the Soft-SVM objective with regularization.
- [ ] Q7a: Derive gradients for output-layer parameters `u` and `c`.
- [ ] Q7b: Derive gradients for hidden-layer parameters `W` and `b`.
- [ ] Q7c: State which forward-pass values must be cached for backpropagation.
- [ ] Q8a: Derive the Jacobian of cross-entropy with respect to probabilities.
- [ ] Q8b: Derive the Jacobian of softmax-cross-entropy with respect to logits.

### C. Required Code Implementations

- [ ] `learning_rate.py`: implement `FixedLR` and `ExponentialLR`.
- [ ] `modules.py`: implement `L1` and squared `L2` objective outputs and gradients/sub-gradients.
- [ ] `gradient_descent.py`: implement GD with:
  - fixed stopping by `max_iter` and tolerance on `||w^(t)-w^(t-1)||`;
  - learning-rate strategy usage;
  - callback arguments exactly as documented;
  - output modes `last`, `best`, and `average`;
  - running average without storing all previous solutions.
- [ ] `gradient_descent_investigation.py`: implement:
  - `get_gd_state_recorder_callback`;
  - `compare_fixed_learning_rates`;
  - `compare_exponential_decay_rates`.
- [ ] `nn_loss_functions.py`: implement numerically stable `softmax` and average `cross_entropy`.
- [ ] `nn_modules.py`: implement:
  - `FullyConnectedLayer`;
  - `ReLU`;
  - `CrossEntropyLoss`;
  - forward caching;
  - backpropagation;
  - `self._grad_weights`;
  - `clear_cache`.
- [ ] `neural_network.py`: implement:
  - constructor;
  - `_fit`;
  - `_loss`;
  - `compute_prediction`;
  - `compute_output`;
  - `compute_jacobian`;
  - `clear_cache`;
  - flattened weight getter/setter compatibility.
- [ ] `stochastic_gradient_descent.py`: implement SGD with:
  - mini-batch sampling;
  - fixed stopping by `max_iter` and tolerance;
  - callback arguments exactly as documented, including `batch_indices`;
  - `_partial_fit`.
- [ ] `nn_utils.py`: implement `confusion_matrix`.

### D. Required Gradient-Descent Experiments and Answers

- [ ] Run fixed-learning-rate GD from `w0 = (sqrt(2), e/3)` for both L1 and L2.
- [ ] Use fixed learning rates `{1, 0.1, 0.01, 0.001}`.
- [ ] Include descent-path plots for `eta = 0.01` in `Answers.pdf`.
- [ ] Explain the L1 vs L2 descent-path differences for `eta = 0.01`.
- [ ] Describe two phenomena observed in L1 descent paths with fixed learning rate.
- [ ] Include convergence-rate plots for L1 and L2 with all fixed learning rates.
- [ ] Explain the convergence-rate plots.
- [ ] Report the lowest loss achieved for L1 and L2 and explain the difference.
- [ ] Run exponential-decay GD on L1 from `w0 = (sqrt(2), e/3)`.
- [ ] Use `eta = 0.1` and `gamma in {0.9, 0.95, 0.99, 1}`.
- [ ] Include one convergence plot containing all decay rates.
- [ ] Explain exponential-decay results compared with fixed learning rate.
- [ ] Report the lowest L1 norm achieved using exponential decay and explain why it differs.
- [ ] Include the descent-path plot for `gamma = 0.95`.
- [ ] Explain how the descent path changed compared with fixed learning rate.

### E. Required Simulated-Data Neural-Network Experiments and Answers

- [ ] In `nn_simulated_data.py`, use the required baseline architecture:
  - two fully connected hidden layers;
  - 16 neurons in each hidden layer;
  - intercepts included and not counted in the 16 neurons;
  - ReLU activations;
  - softmax-cross-entropy loss;
  - implemented Gradient Descent;
  - fixed learning rate `eta = 0.1`;
  - up to `5000` iterations.
- [ ] Q1: Fit the baseline network on the train set.
- [ ] Q1: Include the learned decision-boundary plot in `Answers.pdf`.
- [ ] Q1: Report test accuracy.
- [ ] Q2: Remove both hidden layers only for this question.
- [ ] Q2: Repeat decision-boundary plot and test accuracy.
- [ ] Q2: Explain the result.
- [ ] Q3: Rerun Q1 with a callback recording convergence.
- [ ] Q3: Include loss-vs-iteration and gradient-norm-vs-iteration plots in `Answers.pdf`.
- [ ] Q3: Store network weights every 100 iterations.
- [ ] Q3: Call `animate_decision_boundary` for inspection.
- [ ] Q3: Do not include the animation output in the submission.
- [ ] Q4: Reduce both hidden layers to 6 neurons only for this question.
- [ ] Q4: Repeat Q3 plots and animation.
- [ ] Q4: Explain the result.

### F. Required MNIST Experiments and Answers

- [ ] In `nn_mnist_digit_classification.py`, use the required baseline architecture:
  - two fully connected hidden layers;
  - 64 neurons in each hidden layer;
  - intercepts included and not counted in the 64 neurons;
  - ReLU activations;
  - softmax-cross-entropy loss;
  - implemented SGD;
  - fixed learning rate `eta = 0.1`;
  - up to `10000` iterations;
  - mini-batches of `256`.
- [ ] Add a callback recording current loss and gradient norm.
- [ ] Q5: Train the baseline network and report test accuracy.
- [ ] Q6: Include loss-vs-iteration and gradient-norm-vs-iteration plots in `Answers.pdf`.
- [ ] Q7: Include a confusion matrix for true vs predicted test labels.
- [ ] Q7: Identify the two most common confusions.
- [ ] Q7: Identify the three least common confusions.
- [ ] Q7: Explain whether the confusion results make sense.
- [ ] Q8: Remove both hidden layers only for this question.
- [ ] Q8: Repeat Q5 and explain what the accuracy suggests about the data.
- [ ] Q9: Filter test samples whose true digit is `7`.
- [ ] Q9: Compute prediction confidence as `max_k p_k` after softmax.
- [ ] Q9: Include grids of the 64 most confident and 64 least confident true-7 images.
- [ ] Q9: Explain visible differences between the two image sets.
- [ ] Q10: Train the same baseline architecture twice, once with GD and once with SGD.
- [ ] Q10: Use fixed learning rate `10^-1`, max `10000` iterations, tolerance `10^-10`.
- [ ] Q10: Use SGD batch size `64`.
- [ ] Q10: Record current loss and elapsed time at each iteration.
- [ ] Q10: Train on the first `2500` train samples.
- [ ] Q10: Include GD runtime-vs-loss plot.
- [ ] Q10: Include SGD runtime-vs-loss plot.
- [ ] Q10: Include one combined figure with both solvers as two scatters, no subplots.
- [ ] Q10: Explain similarities and differences in runtime, loss scale, and curve shape.

### G. Final Verification Before Submission

- [ ] Run `python -m compileall -q .` from `ex3/code`.
- [ ] Run `python gradient_descent_investigation.py` from `ex3/code`.
- [ ] Run `python nn_simulated_data.py` from `ex3/code` without `ML_QUICK=1` for final numbers.
- [ ] Run `python nn_mnist_digit_classification.py` from `ex3/code` without `ML_QUICK=1` for final numbers.
- [ ] Inspect all generated plots and copy/embed required plots into `Answers.pdf`.
- [ ] Confirm `Answers.pdf` contains all theoretical answers, practical answers, numeric results, explanations, and required plots.
- [ ] Confirm generated `figures/` files are not treated as substitutes for plots in `Answers.pdf`.
- [ ] Confirm the final zip contains only the required Python files plus `Answers.pdf` at the zip root.

## Progress Notes

- [x] Implemented core GD, SGD, learning rates, L1/L2 modules, NN losses, NN layers, `NeuralNetwork`, and `confusion_matrix`.
- [x] Implemented runnable GD investigation script.
- [x] Implemented runnable simulated-data experiment script.
- [x] Implemented runnable MNIST experiment script.
- [x] Verified GD investigation script with `python gradient_descent_investigation.py`.
- [x] Verified simulated-data script in quick mode with `ML_QUICK=1`.
- [x] Verified MNIST script in quick mode with `ML_QUICK=1`.
- [ ] Run full simulated-data script without quick mode if you want final assignment numbers and plots.
- [ ] Run full MNIST script without quick mode if you want final assignment numbers and plots.
- [ ] Write the theoretical answers and practical explanations in `Answers.pdf`.

## 0. Setup and Orientation

- [ ] Read the general submission instructions on the course website.
- [ ] Install/use the listed Python requirements from `code/requirements`:
  - `numpy`
  - `pandas`
  - `plotly`
  - `imageio`
  - `kaleido`
  - `torch`
  - `torchvision`
- [ ] Work in `ex3/code`.
- [ ] Keep plots and written answers for all theoretical and practical questions in `Answers.pdf`.
- [ ] Use `np.random.seed(0)` where the provided scripts expect reproducibility.

## 1. Theoretical Part: Answers.pdf

### 1.1 Convex Optimization

- [ ] Q1: Prove from the definition that a nonnegative weighted sum of convex functions is convex:
  - Given convex `f_1, ..., f_m : C -> R`.
  - Given weights `g_1, ..., g_m in R_+`.
  - Show `g(u) = sum_i g_i f_i(u)` is convex.
- [ ] Q2: Give a counterexample to the claim:
  - If `f, g : R -> R` are convex, then `h = f o g` is convex.
  - Include the functions, the composed function, and why the composed function is not convex.

### 1.2 Sub-Gradients for Soft-SVM Objective

- [ ] Q3: Prove the hinge loss is convex in `(w, b)`:
  - `l_hinge_{x,y}(w,b) = max(0, 1 - y(x^T w + b))`.
  - Use that the maximum of convex functions is convex.
- [ ] Q4: Derive a valid sub-gradient of the hinge loss with respect to `(w, b)`.
  - Cover the cases margin greater than 1, less than 1, and exactly 1.
- [ ] Q5: Prove that the sum of sub-gradients is a sub-gradient of the sum:
  - If `g_k in partial f_k(x)`, show `sum_k g_k in partial sum_k f_k(x)`.
- [ ] Q6: Derive a sub-gradient for the Soft-SVM objective:
  - `f(w,b) = (1/m) sum_i l_hinge_{x_i,y_i}(w,b) + (lambda/2)||w||^2`.
  - Include the contribution from the regularization term.

### 1.3 Feed-Forward Neural Networks

- [ ] Q7a: For the network `h = ReLU(Wx + b)`, `y_hat = u^T h + c`, `L = 1/2(y_hat - y)^2`, derive gradients with respect to `u` and `c`.
- [ ] Q7b: Derive gradients with respect to hidden layer weights `W` and biases `b`.
- [ ] Q7c: List the forward-pass intermediate values that must be cached for correct backpropagation.
- [ ] Q8a: Derive the Jacobian of cross-entropy loss with respect to probabilities:
  - For one-hot class `i`, `L_i(z) = -log(z_i)`.
  - Find `J_z(L_i)`.
- [ ] Q8b: Apply the chain rule through softmax:
  - Use `J_x(S) = diag(S(x)) - S(x)S(x)^T`.
  - Derive `J_x(L_i o S)`.
  - This result is needed for `CrossEntropyLoss.compute_jacobian`.

## 2. Practical Part: Gradient Descent

### 2.1 Learning Rates

- [ ] Implement `FixedLR.lr_step` in `code/learning_rate.py`.
  - Return the fixed base learning rate.
- [ ] Implement `ExponentialLR.lr_step` in `code/learning_rate.py`.
  - Return `base_lr * decay_rate^t`.

### 2.2 Objective Modules

- [ ] Implement `L2.compute_output` in `code/modules.py`.
  - Function: squared L2 norm, `||w||_2^2`.
- [ ] Implement `L2.compute_jacobian` in `code/modules.py`.
  - Derivative with respect to weights.
- [ ] Implement `L1.compute_output` in `code/modules.py`.
  - Function: L1 norm, `||w||_1`.
- [ ] Implement `L1.compute_jacobian` in `code/modules.py`.
  - Use a valid sub-gradient.
  - Decide and document the value used at zero if needed.

### 2.3 GradientDescent

- [ ] Implement `GradientDescent.fit` in `code/gradient_descent.py`.
- [ ] Support stopping by:
  - `max_iter`.
  - Euclidean norm `||w^(t) - w^(t-1)|| < tol`.
- [ ] Use `self._learning_rate.lr_step(t=t)` each iteration.
- [ ] Update module weights using the module Jacobian.
- [ ] Support all output types:
  - `last`
  - `best`
  - `average`
- [ ] For `average`, maintain a running average without storing all previous solutions.
- [ ] Call the callback at the end of every iteration with:
  - `solver`
  - `weights`
  - `val`
  - `grad`
  - `t`
  - `eta`
  - `delta`

### 2.4 Gradient Descent Investigation

- [ ] Implement `get_gd_state_recorder_callback` in `code/gradient_descent_investigation.py`.
  - Return a fresh callback.
  - Return lists recording objective values and weights.
- [ ] Implement `compare_fixed_learning_rates`.
  - Use initial point `w0 = (sqrt(2), e/3)`.
  - Run both L1 and L2 modules.
  - Use fixed learning rates `{1, 0.1, 0.01, 0.001}`.
- [ ] Produce descent-path plots for each setting.
- [ ] In `Answers.pdf`, include plots for `eta = 0.01`.
- [ ] In `Answers.pdf`, explain differences between L1 and L2 descent paths for `eta = 0.01`.
- [ ] In `Answers.pdf`, describe two phenomena observed in the L1 descent path with fixed learning rate.
- [ ] Plot convergence rate for each module:
  - Norm as a function of GD iteration.
  - Include all specified fixed learning rates.
- [ ] In `Answers.pdf`, explain the convergence-rate plots.
- [ ] In `Answers.pdf`, report the lowest loss achieved for each module and explain the differences.
- [ ] Implement `compare_exponential_decay_rates`.
  - Use L1 objective.
  - Use initial point `w0 = (sqrt(2), e/3)`.
  - Use `eta = 0.1`.
  - Use decay rates `gamma in {0.9, 0.95, 0.99, 1}`.
- [ ] Plot convergence rate for all decay rates in one plot.
- [ ] In `Answers.pdf`, explain results for exponential decay.
- [ ] In `Answers.pdf`, compare exponential decay against fixed learning rate.
- [ ] In `Answers.pdf`, report the lowest L1 norm achieved with exponential decay and explain differences.
- [ ] Plot descent path for `gamma = 0.95`.
- [ ] In `Answers.pdf`, describe how the descent path changed compared with fixed learning rate.

## 3. Practical Part: Neural Network Implementation

### 3.1 Loss Functions

- [ ] Implement `softmax` in `code/nn_loss_functions.py`.
  - Handle batches.
  - Use a numerically stable implementation.
- [ ] Implement `cross_entropy` in `code/nn_loss_functions.py`.
  - Accept labels either as class indices or one-hot vectors.
  - Return average cross-entropy over the batch.

### 3.2 Neural Network Modules

- [ ] Implement `FullyConnectedLayer.__init__` in `code/nn_modules.py`.
  - Store input/output dimensions.
  - Store activation.
  - Store whether an intercept is included.
  - Initialize weights from `N(0, 1/input_dim)`.
  - If intercept is included, account for the extra column of ones in the weight shape.
- [ ] Implement `FullyConnectedLayer.compute_output`.
  - Cache the layer input used in the forward pass.
  - Add intercept column when needed.
  - Compute row-major linear output using `XW`.
  - Apply activation if one is provided.
- [ ] Implement `FullyConnectedLayer.backprop`.
  - If activation exists, backprop through it first.
  - Store weight gradient in `self._grad_weights` using `X^T D`.
  - Return downstream gradient using `D W^T`.
  - If intercept is included, do not propagate the intercept column as an input feature.
- [ ] Implement `FullyConnectedLayer.get_grad_weights`.
- [ ] Implement `FullyConnectedLayer.clear_cache`.
  - Clear cached inputs and gradients after optimizer has collected them.
- [ ] Implement `ReLU.compute_output`.
  - Cache activation input.
  - Return elementwise `max(X, 0)`.
- [ ] Implement `ReLU.backprop`.
  - Multiply upstream gradient by the elementwise ReLU derivative.
- [ ] Implement `ReLU.clear_cache`.
- [ ] Implement `CrossEntropyLoss.compute_output`.
  - Apply softmax to logits.
  - Return average cross-entropy loss.
- [ ] Implement `CrossEntropyLoss.compute_jacobian`.
  - Use the softmax-cross-entropy derivative from the theoretical part.
  - Support batches.
  - Return shape `(n_samples, input_dim)`.

### 3.3 NeuralNetwork Class

- [ ] Implement `NeuralNetwork.__init__` in `code/neural_network.py`.
  - Store modules, loss function, and solver.
- [ ] Implement `_fit`.
  - Fit the network over data using the solver.
- [ ] Implement `_loss`.
  - Compute the network loss over given data.
- [ ] Implement `compute_prediction`.
  - Forward pass through all network layers.
  - Return logits before loss and before softmax.
- [ ] Implement `compute_output`.
  - Compute logits.
  - Compute and return loss using the loss function.
  - Cache whatever is needed for backward pass.
- [ ] Implement `compute_jacobian`.
  - Start with `loss_fn.compute_jacobian(logits, y)`.
  - Traverse layers in reverse order.
  - Call each layer's `backprop`.
  - Collect each layer's `get_grad_weights`.
  - Flatten gradients into one vector with `_flatten_parameters`.
  - Call `clear_cache` appropriately after gradients are collected.
- [ ] Implement `clear_cache`.
  - Clear network and layer caches.
- [ ] Verify the `weights` getter/setter works with flattened parameters.

### 3.4 Stochastic Gradient Descent

- [ ] Implement `StochasticGradientDescent.__init__` in `code/stochastic_gradient_descent.py`.
  - Store learning rate, tolerance, max iterations, batch size, and callback.
- [ ] Implement `StochasticGradientDescent.fit`.
  - Sample random mini-batches each iteration.
  - Stop by `max_iter` or `delta < tol`.
  - Return final solution.
  - Call callback every iteration with all GD callback fields plus `batch_indices`.
- [ ] Implement `StochasticGradientDescent._partial_fit`.
  - Compute batch loss.
  - Compute batch gradient.
  - Compute current learning rate.
  - Apply one SGD update.
  - Return `(val, jac, eta)`.

### 3.5 Utilities

- [ ] Implement `confusion_matrix` in `code/nn_utils.py`.
  - Rows correspond to unique values in the first vector.
  - Columns correspond to unique values in the second vector.
  - Entry `(i, j)` counts how often value `i` in first vector appears with value `j` in second vector.

## 4. Practical Part: Simulated 2D Dataset

File: `code/nn_simulated_data.py`

- [ ] Build the required baseline architecture:
  - Two fully connected hidden layers.
  - Each hidden layer has 16 neurons.
  - Include intercepts, not counted in the 16 neurons.
  - Use ReLU activations.
  - Use softmax-cross-entropy loss.
  - Use implemented Gradient Descent.
  - Use fixed learning rate `eta = 0.1`.
  - Use up to `5000` iterations.
- [ ] Q1: Fit on the training set.
- [ ] Q1: Plot the learned decision boundary using `plot_decision_boundary`.
- [ ] Q1: Evaluate and report test accuracy.
- [ ] Q2: Remove both hidden layers only for this question.
- [ ] Q2: Repeat fitting, decision-boundary plot, and test accuracy.
- [ ] Q2: Explain the results.
- [ ] Q3: Rerun the Q1 network with a callback that records convergence.
- [ ] Q3: Plot loss as a function of iteration.
- [ ] Q3: Plot gradient norm as a function of iteration.
- [ ] Q3: Store network weights every 100 iterations.
- [ ] Q3: Call `animate_decision_boundary` to inspect boundary evolution.
- [ ] Q3: Do not include the animation output in the submission.
- [ ] Q4: Decrease hidden-layer width to 6 neurons each only for this question.
- [ ] Q4: Repeat Q3 plots and animation.
- [ ] Q4: Explain the results.

## 5. Practical Part: MNIST Classification

File: `code/nn_mnist_digit_classification.py`

### 5.1 Baseline MNIST Network

- [ ] Build the required baseline architecture:
  - Two fully connected hidden layers.
  - Each hidden layer has 64 neurons.
  - Include intercepts, not counted in the 64 neurons.
  - Use ReLU activations.
  - Use softmax-cross-entropy loss.
  - Use implemented SGD.
  - Use fixed learning rate `eta = 0.1`.
  - Use up to `10000` iterations.
  - Use mini-batches of `256` samples.
- [ ] Add a callback that records:
  - Current loss.
  - Gradient norm.

### 5.2 MNIST Questions

- [ ] Q5: Train the neural network on the MNIST train set.
- [ ] Q5: Evaluate and report test accuracy.
- [ ] Q6: Plot convergence:
  - Loss as a function of iteration.
  - Gradient norm as a function of iteration.
- [ ] Q7: Plot a confusion matrix between true and predicted test labels.
- [ ] Q7: Identify the two most common confusions.
- [ ] Q7: Identify the three least common confusions.
- [ ] Q7: Explain whether the confusion results make sense.
- [ ] Q8: Remove both hidden layers only for this question.
- [ ] Q8: Repeat Q5.
- [ ] Q8: Explain what the test accuracy suggests about the data.
- [ ] Q9: Filter the test set to samples whose true digit is `7`.
- [ ] Q9: Compute prediction confidence as `max_k p_k` after softmax.
- [ ] Q9: Plot the 64 most confident digit-7 images using `plot_images_grid`.
- [ ] Q9: Plot the 64 least confident digit-7 images using `plot_images_grid`.
- [ ] Q9: Explain visible differences between the most and least confident sets.
- [ ] Q10: Compare runtime behavior of GD and SGD using the same baseline MNIST architecture.
- [ ] Q10: Initialize both solvers with:
  - Fixed learning rate `10^-1`.
  - Maximum `10000` iterations.
  - Tolerance `10^-10`.
- [ ] Q10: For SGD, use batch size `64`.
- [ ] Q10: Add a callback that records at each iteration:
  - Network current loss.
  - Time elapsed from fit start.
- [ ] Q10: Train both networks on the first `2500` training samples.
- [ ] Q10: Plot runtime vs loss for GD alone.
- [ ] Q10: Plot runtime vs loss for SGD alone.
- [ ] Q10: Plot both solvers on one figure as two different scatters, no subplots.
- [ ] Q10: Explain similarities and differences:
  - Running times.
  - Loss scales.
  - Curve shapes.

## 6. Verification Checklist

- [ ] Run the gradient descent investigation script and confirm it completes.
- [ ] Run simulated-data neural network experiments and confirm all requested plots can be generated.
- [ ] Run MNIST experiments and confirm all requested plots can be generated.
- [ ] Check that the loss generally decreases or behaves plausibly.
- [ ] Check that gradients have expected shapes.
- [ ] Check that flattened network gradients match flattened network weights in size.
- [ ] Check that callbacks receive all documented keyword arguments.
- [ ] Check that caches are cleared after backward passes to avoid memory buildup.
- [ ] Check that all required plots are embedded inside `Answers.pdf`.
- [ ] Check that no required answer is left only in code comments or separate plot files.

## 7. Submission Requirements Notes

- [ ] Submit one zip file named `ex3_ID.zip`, replacing `ID` with your ID.
- [ ] Submit the zip in the designated Moodle activity before the Moodle deadline.
- [ ] The zip must contain `Answers.pdf`.
- [ ] `Answers.pdf` must include:
  - Answers to all theoretical questions.
  - Answers to all practical questions.
  - All required plotted graphs.
- [ ] Do not submit plots only as separate image files.
  - The PDF says plots included as separate files will be considered not provided.
- [ ] The zip must contain the following Python files directly inside the zip, without directories:
  - `gradient_descent.py`
  - `gradient_descent_investigation.py`
  - `learning_rate.py`
  - `modules.py`
  - `neural_network.py`
  - `nn_loss_functions.py`
  - `nn_mnist_digit_classification.py`
  - `nn_simulated_data.py`
  - `nn_utils.py`
  - `nn_modules.py`
  - `stochastic_gradient_descent.py`
- [ ] Do not rely on the folder structure in the zip; the required files should be at the zip root.
- [ ] Before zipping, verify every required file above is present and contains your final implementation.
