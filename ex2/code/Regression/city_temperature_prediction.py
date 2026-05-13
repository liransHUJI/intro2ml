import os

import random
import numpy as np
import pandas as pd
import plotly.express as px

from polynomial_fitting import PolynomialFitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Use parse_dates to ensure Date column parsed as datetime
    data = pd.read_csv(filename, parse_dates=["Date"], dayfirst=False)

    # Drop rows with missing critical fields
    data = data.dropna(subset=["Date", "Temp", "Country", "Year", "Month", "Day"])

    # Derive DayOfYear from Date column
    data["DayOfYear"] = pd.to_datetime(data["Date"]).dt.dayofyear

    # Filter unrealistic temperature values
    mask = (data["Temp"] > -50) & (data["Temp"] < 55)
    data = data.loc[mask].reset_index(drop=True)

    return data


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    path = os.path.dirname(os.path.abspath(__file__))
    data = load_data(f"{path}/city_temperature.csv")
        
    
    # Question 3 - Exploring data for specific country
    data_il = data[data["Country"] == "Israel"].copy()
    data_il["Year"] = data_il["Year"].astype(str)
    fig = px.scatter(
        data_frame=data_il,
        x="DayOfYear",  # Feature to be used for polynomial fitting [cite: 110]
        y="Temp",       # Response variable 
        color="Year",   # Discrete color coding by different years 
        labels={"DayOfYear": "Day of Year", "Temp": "Average Daily Temperature"},
        title="Average Daily Temperature in Israel vs. Day of Year"
    )
    fig.show()

    # 1. Group by Month and calculate Standard Deviation
    # We use 'agg' to specifically target the Temp column
    monthly_std = data_il.groupby("Month")["Temp"].agg("std").reset_index()
    monthly_std.columns = ["Month", "Standard Deviation"]  # Rename columns for clarity
    fig = px.bar(
        data_frame=monthly_std,
        x="Month",
        y="Standard Deviation",
        title="Standard Deviation of Daily Temperatures per Month (Israel)",
        labels={"Standard Deviation": "Temperature Std Deviation", "Month": "Month"},
        color="Standard Deviation", # Adding color makes the variance easier to see
    )
    # Ensure all month numbers (1-12) are shown on the x-axis
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1))

    fig.show()
    
    # Question 4 - Fitting model for different values of `k` (Israel-only)
    seed = 42

    # Filter to Israel subset for analysis
    data_il = data[data["Country"] == "Israel"].reset_index(drop=True)
    if data_il.empty:
        raise RuntimeError("No Israel data found in dataset.")

    # Shuffle Israel data and split 75/25
    data_il = data_il.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(0.75 * len(data_il))
    train_data = data_il.iloc[:split_idx]
    test_data = data_il.iloc[split_idx:]

    X_train = train_data["DayOfYear"].to_numpy()
    y_train = train_data["Temp"].to_numpy()
    X_test = test_data["DayOfYear"].to_numpy()
    y_test = test_data["Temp"].to_numpy()

    errors = []
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(X_train, y_train)

        test_loss = poly_fit.loss(X_test, y_test)
        test_loss_rounded = round(test_loss, 2)
        errors.append(test_loss_rounded)
        print(f"Loss for k={k}: {test_loss_rounded}")

    # Choose best k (smallest loss, tie -> smallest k)
    min_loss = min(errors)
    best_ks = [i + 1 for i, v in enumerate(errors) if v == min_loss]
    best_k = min(best_ks)
    print(f"Best k selected: {best_k} with test loss {min_loss}")

    # Bar plot of errors per k
    plot_data = pd.DataFrame({"k": range(1, 11), "Loss": errors})
    fig = px.bar(
        data_frame=plot_data,
        x="k",
        y="Loss",
        title="MSE Loss of Polynomial Fitting for Different Degrees (k) (Israel Test Set)",
        labels={"Loss": "Mean Squared Error", "k": "Degree of Polynomial"},
        color="Loss",
    )
    fig.show()
