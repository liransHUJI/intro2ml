import time
import numpy as np
import gzip
import os
from pathlib import Path
from typing import Tuple


from nn_loss_functions import accuracy
from nn_modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax
from neural_network import NeuralNetwork
from gradient_descent import GradientDescent  # use implementation from previous exercise
from learning_rate import FixedLR  # use implementation from previous exercise
from stochastic_gradient_descent import StochasticGradientDescent
from nn_utils import confusion_matrix


import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('mnist-train-images.gz'),
            load_labels('mnist-train-labels.gz'),
            load_images('mnist-test-images.gz'),
            load_labels('mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray", )\
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False)\
        .update_xaxes(showticklabels=False)\
        .update_yaxes(showticklabels=False)


if __name__ == '__main__':
    np.random.seed(0)
    Path("figures").mkdir(exist_ok=True)
    quick = os.environ.get("ML_QUICK") == "1"
    train_X, train_y, test_X, test_y = load_mnist()
    if quick:
        train_X, train_y = train_X[:1500], train_y[:1500]
        test_X, test_y = test_X[:500], test_y[:500]
    (n_samples, n_features), n_classes = train_X.shape, 10

    def make_baseline_network(solver, hidden_width=64):
        return NeuralNetwork(
            modules=[
                FullyConnectedLayer(n_features, hidden_width, ReLU()),
                FullyConnectedLayer(hidden_width, hidden_width, ReLU()),
                FullyConnectedLayer(hidden_width, n_classes),
            ],
            loss_fn=CrossEntropyLoss(),
            solver=solver,
        )

    def make_linear_network(solver):
        return NeuralNetwork(
            modules=[FullyConnectedLayer(n_features, n_classes)],
            loss_fn=CrossEntropyLoss(),
            solver=solver,
        )

    def convergence_callback_factory():
        losses, grad_norms = [], []

        def callback(**kwargs):
            losses.append(float(np.squeeze(kwargs["val"])))
            grad_norms.append(float(np.linalg.norm(kwargs["grad"])))

        return callback, losses, grad_norms

    sgd_iterations = 300 if quick else 10000

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    callback, losses, grad_norms = convergence_callback_factory()
    network = make_baseline_network(
        StochasticGradientDescent(
            learning_rate=FixedLR(.1),
            max_iter=sgd_iterations,
            batch_size=256,
            callback=callback,
        )
    ).fit(train_X, train_y)
    test_pred = network.predict(test_X)
    q5_accuracy = accuracy(test_y, test_pred)
    print(f"Q5 test accuracy: {q5_accuracy:.4f}")

    make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Gradient Norm"))\
        .add_trace(go.Scatter(y=losses, mode="lines"), row=1, col=1)\
        .add_trace(go.Scatter(y=grad_norms, mode="lines"), row=1, col=2)\
        .update_layout(title="MNIST Q6 Convergence", showlegend=False)\
        .write_image("figures/mnist_q6_convergence.png")

    cm = confusion_matrix(test_y, test_pred)
    true_labels = np.unique(test_y)
    pred_labels = np.unique(test_pred)
    ff.create_annotated_heatmap(cm, x=pred_labels.tolist(), y=true_labels.tolist(), colorscale="Blues")\
        .update_layout(title="MNIST Q7 Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")\
        .write_image("figures/mnist_q7_confusion_matrix.png")

    off_diagonal = cm.copy()
    for i, true_label in enumerate(true_labels):
        matches = np.where(pred_labels == true_label)[0]
        if matches.size:
            off_diagonal[i, matches[0]] = 0
    flat_order = np.argsort(off_diagonal.ravel())
    nonzero_pairs = [
        (
            int(true_labels[idx // cm.shape[1]]),
            int(pred_labels[idx % cm.shape[1]]),
            int(off_diagonal.ravel()[idx]),
        )
        for idx in flat_order
        if off_diagonal.ravel()[idx] > 0
    ]
    common_pairs = list(reversed(nonzero_pairs[-2:]))
    least_pairs = nonzero_pairs[:3]
    print(f"Q7 two most common off-diagonal confusions: {common_pairs}")
    print(f"Q7 three least common nonzero off-diagonal confusions: {least_pairs}")

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    linear_network = make_linear_network(
        StochasticGradientDescent(
            learning_rate=FixedLR(.1),
            max_iter=sgd_iterations,
            batch_size=256,
        )
    ).fit(train_X, train_y)
    q8_accuracy = accuracy(test_y, linear_network.predict(test_X))
    print(f"Q8 no-hidden-layers test accuracy: {q8_accuracy:.4f}")

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    digit = 7
    digit_mask = test_y == digit
    digit_X = test_X[digit_mask]
    digit_logits = network.compute_prediction(digit_X)
    digit_probabilities = softmax(digit_logits)
    confidences = np.max(digit_probabilities, axis=1)
    side = int(np.sqrt(min(64, digit_X.shape[0])))
    n_show = side ** 2
    most_confident = np.argsort(confidences)[-n_show:]
    least_confident = np.argsort(confidences)[:n_show]
    plot_images_grid(digit_X[most_confident], f"{n_show} most confident true-{digit} predictions")\
        .write_image("figures/mnist_q9_most_confident_7s.png")
    plot_images_grid(digit_X[least_confident], f"{n_show} least confident true-{digit} predictions")\
        .write_image("figures/mnist_q9_least_confident_7s.png")
    print(f"Q9 digit-{digit} confidence range: min={confidences.min():.4f}, max={confidences.max():.4f}")

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    q10_samples = 500 if quick else 2500
    q10_iterations = 100 if quick else 10000
    q10_train_X, q10_train_y = train_X[:q10_samples], train_y[:q10_samples]

    def timing_callback_factory():
        start = time.time()
        losses, times = [], []

        def callback(**kwargs):
            losses.append(float(np.squeeze(kwargs["val"])))
            times.append(time.time() - start)

        return callback, losses, times

    gd_callback, gd_losses, gd_times = timing_callback_factory()
    gd_network = make_baseline_network(
        GradientDescent(
            learning_rate=FixedLR(1e-1),
            max_iter=q10_iterations,
            tol=1e-10,
            callback=gd_callback,
        )
    )
    gd_network.fit(q10_train_X, q10_train_y)

    sgd_callback, sgd_losses, sgd_times = timing_callback_factory()
    sgd_network = make_baseline_network(
        StochasticGradientDescent(
            learning_rate=FixedLR(1e-1),
            max_iter=q10_iterations,
            tol=1e-10,
            batch_size=64,
            callback=sgd_callback,
        )
    )
    sgd_network.fit(q10_train_X, q10_train_y)

    go.Figure(go.Scatter(x=gd_times, y=gd_losses, mode="markers"))\
        .update_layout(title="MNIST Q10 GD runtime vs loss", xaxis_title="Seconds", yaxis_title="Loss")\
        .write_image("figures/mnist_q10_gd_runtime_loss.png")
    go.Figure(go.Scatter(x=sgd_times, y=sgd_losses, mode="markers"))\
        .update_layout(title="MNIST Q10 SGD runtime vs loss", xaxis_title="Seconds", yaxis_title="Loss")\
        .write_image("figures/mnist_q10_sgd_runtime_loss.png")
    go.Figure([
        go.Scatter(x=gd_times, y=gd_losses, mode="markers", name="GD"),
        go.Scatter(x=sgd_times, y=sgd_losses, mode="markers", name="SGD"),
    ])\
        .update_layout(title="MNIST Q10 GD vs SGD runtime vs loss", xaxis_title="Seconds", yaxis_title="Loss")\
        .write_image("figures/mnist_q10_gd_vs_sgd_runtime_loss.png")
    print(f"Q10 GD final loss: {gd_losses[-1]:.4f}, elapsed: {gd_times[-1]:.2f}s")
    print(f"Q10 SGD final loss: {sgd_losses[-1]:.4f}, elapsed: {sgd_times[-1]:.2f}s")
