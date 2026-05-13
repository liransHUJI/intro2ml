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
    data = pd.read_csv(filename)
    data.dropna(inplace=True) # Remove rows with missing values

    data["DayOfYear"] = pd.to_datetime(data[["Year", "Month", "Day"]]).dt.dayofyear 
    data = data.drop(columns=["Date"])
    mask = (data["Temp"] > -50) & (data["Temp"] < 55)
    data = data[mask]
    return data
    pass


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
    
    # Question 4 - Fitting model for different values of `k`
    seed = 42 # random.randint(0, 1000)
    # Split data into 75% train and 25% test
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(0.75 * len(data))
    
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    y = np.array(data["Temp"].to_numpy())
    X = np.array(data["DayOfYear"].to_numpy())
    loss = []
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(X, y)
        loss.append(round(poly_fit.loss(X, y), 2))
        print(f"Loss for k={k}: {loss[-1]}")

    plot_data = pd.DataFrame({"k": range(1, 11), "Loss": loss})
    fig = px.bar(
        data_frame=plot_data,
        x="k",
        y="Loss",
        title="MSE Loss of Polynomial Fitting for Different Degrees (k)",
        labels={"Loss": "Mean Squared Error", "k": "Degree of Polynomial"},
        color="Loss", # Adding color makes it easier to see which k has the lowest loss
    )
    fig.show()
    pass
