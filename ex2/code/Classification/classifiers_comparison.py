import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import accuracy_score


def split_train_test_np(X: np.ndarray, y: np.ndarray, train_proportion: float = 0.8, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    train_n = int(np.floor(train_proportion * n))
    train_idx = perm[:train_n]
    test_idx = perm[train_n:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def to_numpy_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.astype(float), y.astype(int)





### HELPER FUNCTIONS ###
# Add here any helper functions which you think will be useful


def make_two_gaussians(n_samples: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    cov = np.array([[0.5, 0.2], 
                    [0.2, 0.5]])
    half = n_samples // 2

    x0 = rng.multivariate_normal(mean=[-1, -1], cov=cov, size=half)
    x1 = rng.multivariate_normal(mean=[1, 1], cov=cov, size=n_samples - half)

    x = np.vstack([x0, x1])
    y = np.hstack([
        np.zeros(half, dtype=int),
        np.ones(n_samples - half, dtype=int),
    ])

    return x, y


def to_numpy_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.astype(float), y.astype(int)


def lighten_color(color, amount: float = 0.55):
    rgb = np.array(to_rgb(color))
    return tuple(1 - amount * (1 - rgb))


def make_light_cmap(base_colors):
    return ListedColormap([lighten_color(color) for color in base_colors])


def fit_models(train_X, train_y):
    return {
        "SVM (C=1/5)": SVC(kernel="linear", C=1 / 5),
        "Decision Tree (depth=7)": DecisionTreeClassifier(max_depth=7, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }


def plot_boundary(ax, model, X_ref: np.ndarray, test_X: np.ndarray, test_y: np.ndarray, title: str):
    base_colors = ["#1f77b4", "#ff7f0e"]
    light_cmap = make_light_cmap(base_colors)

    padding = 0.6
    x_min, x_max = X_ref[:, 0].min() - padding, X_ref[:, 0].max() + padding
    y_min, y_max = X_ref[:, 1].min() - padding, X_ref[:, 1].max() + padding

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    pred = model.predict(grid).reshape(grid_x.shape)

    ax.contourf(grid_x, grid_y, pred, levels=[-0.5, 0.5, 1.5], cmap=light_cmap, alpha=0.9)

    # Plot only test samples as required
    for label, color in zip([0, 1], base_colors):
        mask = test_y == label
        ax.scatter(
            test_X[mask, 0],
            test_X[mask, 1],
            c=[color],
            edgecolors="white",
            linewidths=0.6,
            s=35,
            label=f"Class {label}",
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def compare_on_dataset(dataset_name: str, X: np.ndarray, y: np.ndarray):
    X_np, y_np = to_numpy_inputs(X, y)
    train_X, train_y, test_X, test_y = split_train_test_np(X_np, y_np, train_proportion=0.8, seed=42)

    models = fit_models(train_X, train_y)
    scores = {}

    for model_name, model in models.items():
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        scores[model_name] = accuracy_score(test_y, pred)

    return train_X, train_y, test_X, test_y, models, scores


### Exercise Solution ###


if __name__ == "__main__":
    # set random seed for reproducibility
    seed = 484652
    # Generate 3 different datasets - Moons, Circles and Two-Gaussians
    X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=seed)
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=seed)
    X_gaussians, y_gaussians = make_two_gaussians(n_samples=200, seed=seed)

    datasets = {
        "Moons": (X_moons, y_moons),
        "Circles": (X_circles, y_circles),
        "Two Gaussians": (X_gaussians, y_gaussians),
    }

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 14), constrained_layout=True)

    for row_idx, (dataset_name, (X, y)) in enumerate(datasets.items()):
        X_np, y_np = to_numpy_inputs(X, y)
        train_X, train_y, test_X, test_y = split_train_test_np(X_np, y_np, train_proportion=0.8, seed=seed)

        models = fit_models(train_X, train_y)

        for col_idx, (model_name, model) in enumerate(models.items()):
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            accuracy = accuracy_score(test_y, predictions)

            ax = axes[row_idx, col_idx]
            plot_boundary(
                ax,
                model,
                X_np,
                test_X,
                test_y,
                title=f"{dataset_name} | {model_name}\nTest accuracy: {accuracy:.2f}",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Classification Boundaries and Model Comparison", fontsize=16)
    plt.show()

    # Generate decision boundary plots
    






