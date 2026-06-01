import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from modules import L1, L2

from gradient_descent import GradientDescent
from learning_rate import FixedLR, ExponentialLR

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from nn_presentation_utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def callback(**kwargs):
        values.append(np.copy(kwargs["val"]))
        weights.append(np.copy(kwargs["weights"]))

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # Optimize L1 and L2 modules using Gradient Descent for each eta
    Path("figures").mkdir(exist_ok=True)
    results = {}
    for module in (L1, L2):
        module_name = module.__name__
        results[module_name] = {}
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            objective = module(weights=np.copy(init))
            solver = GradientDescent(learning_rate=FixedLR(eta), max_iter=1000, callback=callback)
            solver.fit(objective, X=np.empty((0, 0)), y=np.empty(0))
            results[module_name][eta] = {
                "values": np.asarray(values).reshape(-1),
                "weights": np.asarray(weights),
                "final_weights": np.copy(objective.weights),
            }

    # Plot descent path for each setting and observe L1 vs L2 differences
    for module in (L1, L2):
        module_name = module.__name__
        for eta in etas:
            fig = plot_descent_path(
                module,
                results[module_name][eta]["weights"],
                title=f"{module_name}, eta={eta}",
                xrange=(-2, 2),
                yrange=(-2, 2),
            )
            fig.write_html(f"figures/{module_name}_fixed_eta_{eta}_descent_path.html")

    # Plot convergence rate (norm vs iteration) for all learning rates
    for module in (L1, L2):
        module_name = module.__name__
        fig = go.Figure()
        for eta in etas:
            norms = np.linalg.norm(results[module_name][eta]["weights"], axis=1)
            fig.add_trace(go.Scatter(y=norms, mode="lines", name=f"eta={eta}"))
        fig.update_layout(
            title=f"{module_name} GD convergence with fixed learning rates",
            xaxis_title="Iteration",
            yaxis_title="Norm of weights",
        )
        fig.write_html(f"figures/{module_name}_fixed_learning_rates_convergence.html")

    return results


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    Path("figures").mkdir(exist_ok=True)
    results = {}
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        objective = L1(weights=np.copy(init))
        solver = GradientDescent(learning_rate=ExponentialLR(eta, gamma), max_iter=1000, callback=callback)
        solver.fit(objective, X=np.empty((0, 0)), y=np.empty(0))
        results[gamma] = {
            "values": np.asarray(values).reshape(-1),
            "weights": np.asarray(weights),
            "final_weights": np.copy(objective.weights),
        }

    # Plot algorithm's convergence for the different values of gamma
    fig = go.Figure()
    for gamma in gammas:
        norms = np.linalg.norm(results[gamma]["weights"], ord=1, axis=1)
        fig.add_trace(go.Scatter(y=norms, mode="lines", name=f"gamma={gamma}"))
    fig.update_layout(
        title="L1 GD convergence with exponential learning-rate decay",
        xaxis_title="Iteration",
        yaxis_title="L1 norm",
    )
    fig.write_html("figures/L1_exponential_decay_convergence.html")

    # Plot descent path for gamma=0.95
    if .95 in results:
        fig = plot_descent_path(
            L1,
            results[.95]["weights"],
            title=f"L1, eta={eta}, gamma=0.95",
            xrange=(-2, 2),
            yrange=(-2, 2),
        )
        fig.write_html("figures/L1_exponential_gamma_0.95_descent_path.html")

    return results


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
