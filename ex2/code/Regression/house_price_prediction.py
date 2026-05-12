"""
House Price Prediction with Linear Regression - Learning Curve Analysis
========================================================================
Analyzes model performance as training set size increases.
"""

import os
import random

import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

from linear_regression import LinearRegression as LR


# ============================================================================
# PREPROCESSING: Training Data
# ============================================================================

def preprocess_train(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    # ---- Step 1: Drop unnecessary columns ----
    X = X.drop(columns=["id"]).copy()
    X = X.drop(columns=["date"])
    
    # ---- Step 2: Engineer new features ----
    X["is_5_bedroom"] = (X["bedrooms"] == 5).astype(int)
    X["yr_old_house"] = 2026 - X["yr_built"]
    X["yrs_since_renovation"] = 2026 - X["yr_renovated"]

    # ---- Step 3: Filter to valid samples (realistic ranges) ----
    valid_mask = (
        (X["bedrooms"] > 0) &
        (X["bedrooms"] < 10) &
        (X["bathrooms"] > 0) &
        (X["sqft_living"] > 0) &
        (X["sqft_living"] < 12000) &
        (X["sqft_lot"] > 0) &
        (X["floors"] > 0) &
        (X["waterfront"] >= 0) &
        (X["view"] >= 0) &
        (X["condition"] > 0) &
        (X["grade"] > 0) &
        (X["sqft_above"] > 0) &
        (X["sqft_basement"] >= 0) &
        (X["sqft_basement"] < 3500) &
        (X["yr_built"] >= 1500) &
        (X["yr_renovated"] >= 0) &
        (X["sqft_living15"] > 0) &
        (X["sqft_lot15"] > 0) &
        (X["sqft_lot15"] < 350000) &
        (y > 0)
    )
    
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    # ---- Step 4: Handle missing values ----
    if X.isnull().any().any():
        X = X.fillna(X.mean())
        print("removed rows with NaN in X")
    if y.isnull().any():
        valid_y_idx = y.dropna().index
        y = y.loc[valid_y_idx]
        X = X.loc[valid_y_idx]  # Keep X and y aligned
        print("removed rows with NaN in y")
    
    # ---- Step 5: Validate consistency ----
    if len(X) != len(y):
        raise ValueError("Mismatch in number of samples between X and y after preprocessing.")
    if X.isnull().any().any():
        raise ValueError(f"Training set contains NaN values after preprocessing. Columns with NaN: {X.columns[X.isnull().any()].tolist()}")
    if y.isnull().any():
        raise ValueError("Training labels contain NaN values after preprocessing.")
    
    return X, y


# ============================================================================
# PREPROCESSING: Test Data
# ============================================================================

def preprocess_test(X: pd.DataFrame) -> pd.DataFrame:
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    # ---- Step 1: Drop unnecessary columns ----
    X = X.drop(columns=["id"])
    X = X.drop(columns=["date"])

    # ---- Step 2: Engineer same features as training ----
    X["is_5_bedroom"] = (X["bedrooms"] == 5).astype(int)
    X["yr_old_house"] = 2026 - X["yr_built"]
    X["yrs_since_renovation"] = 2026 - X["yr_renovated"]

    # ---- Step 3: Handle missing values (imputation only, no row removal) ----
    if X.isnull().any().any():
        X = X.fillna(X.mean())
    
    # ---- Step 4: Validate no NaN remain ----
    if X.isnull().any().any():
        raise ValueError(f"Test set contains NaN values after preprocessing. Columns with NaN: {X.columns[X.isnull().any()].tolist()}")
    
    return X


# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> None:
    # Ensure the output directory exists 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Generate correlation plot for each feature
    for feature_name in X.columns:
        # Compute Pearson correlation coefficient
        correlation = X[feature_name].corr(y)
        
        # Create scatter plot
        fig = px.scatter(
            x=X[feature_name], 
            y=y,
            labels={"x": f"{feature_name}", "y": "Response (Price)"},
            title=f"Feature: {feature_name} | Pearson Correlation: {correlation:.4f}"
        )

        # Save plot to disk
        file_path = os.path.join(output_path, f"correlation_{feature_name}.png")
        fig.write_image(file_path)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == '__main__':
    # ---- SETUP ----
    path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(f"{path}/house_prices.csv")
    X = df.drop("price", axis=1)
    
    # Convert scientific-notation price values (e.g. 1.199e+006) to integers
    y = pd.to_numeric(df["price"], errors="coerce").round()
    valid_price = y.notna()
    X = X.loc[valid_price].reset_index(drop=True)
    y = y.loc[valid_price].astype(np.int64).reset_index(drop=True)
    
    # Question 2 - split train test
    seed = random.randint(0, 1000)
    perm = X.sample(frac=1, random_state=seed).index
    X = X.loc[perm].reset_index(drop=True)
    y = y.loc[perm].reset_index(drop=True)

    train_X = X.iloc[:int(0.75 * len(X))]
    train_y = y.iloc[:int(0.75 * len(y))]
    test_X = X.iloc[int(0.75 * len(X)):]
    test_y = y.iloc[int(0.75 * len(y)):]

    # Question 3 - preprocessing of housing prices train dataset
    train_X, train_y = preprocess_train(train_X, train_y)

    # Question 4 - preprocess the test data
    test_X = preprocess_test(test_X) # 

    # Question 5 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean   -2*std, mean+2*std)

    mean_losses = []
    std_losses = []
    percentages = np.arange(10, 101)
    feature_evaluation(train_X, train_y, output_path=f"{path}/correlations")
    for p in percentages:
        losses = []
        
        # Run 10 independent trials for this percentage
        for _ in range(10):
            sample_X = train_X.sample(frac=p / 100.0)
            sample_y = train_y.loc[sample_X.index]
            
            # Fit model on sampled training data
            model = LR(include_intercept=True)
            model.fit(sample_X.to_numpy(), sample_y.to_numpy())

            # Evaluate on static test set
            loss = model.loss(test_X.to_numpy(), test_y.to_numpy())
            losses.append(loss)
            
        # Store mean and std of losses for this percentage
        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))
        
    # ---- Plot Learning Curve ----
    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)
    
    plt.figure()
    plt.plot(percentages, mean_losses, label="Mean Loss")
    plt.fill_between(percentages, 
                     mean_losses - 2 * std_losses, 
                     mean_losses + 2 * std_losses, 
                     alpha=0.3, label="Confidence Interval")
    plt.xlabel("Percentage of training data (%)")
    plt.ylabel("Mean Squared Error over Test Set")
    plt.title("MSE vs. Training Data Size")
    plt.legend()
    plt.show()

