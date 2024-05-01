import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

def half_life(w, features):
    l = np.exp(features[:, 1] * w[1] + features[:, 2] * w[2] + features[:, 3] * w[3] + w[4])
    return np.exp(-features[:, 0] / l)

def loss_func(w, features, ground_truth):
    return log_loss(ground_truth, half_life(w, features))

for train_filename in ["train.csv", "train_extra.csv"]:
    for test_filename in ["test.csv", "test_extra.csv"]:
        features = ["elapsedDays_{t-1}", "correctSum_{1:t-1}", "incorrectSum_{1:t-1}", "ease_{t-1}"]

        ####################
        # Train model
        ####################

        train_df = pd.read_csv(train_filename)

        df_features = train_df[features].values
        df_ground_truth = train_df["recall_{t}"].values

        np.random.seed(3)
        initial_w = np.random.randn(6) * 1e-3

        result = minimize(loss_func, initial_w, args=(df_features, df_ground_truth))

        ####################
        # Test model
        ####################

        test_df = pd.read_csv(test_filename)

        df_features = test_df[features].values
        df_ground_truth = test_df["recall_{t}"]

        w = result.x
        pred = half_life(w, df_features)

        low_range = np.quantile(test_df["elapsedDays_{t}"], q=0.3)
        mid_range = np.quantile(test_df["elapsedDays_{t}"], q=0.6)

        low_mask = test_df["elapsedDays_{t}"] <= low_range
        pred0 = pred[low_mask]
        gt0 = df_ground_truth[low_mask]
        loss0 = np.mean((pred0 - gt0) ** 2)

        mid_mask = (test_df["elapsedDays_{t}"] > low_range) & (test_df["elapsedDays_{t}"] <= mid_range)
        pred1 = pred[mid_mask]
        gt1 = df_ground_truth[mid_mask]
        loss1 = np.mean((pred1 - gt1) ** 2)

        long_mask = test_df["elapsedDays_{t}"] > mid_range
        pred2 = pred[long_mask]
        gt2 = df_ground_truth[long_mask]
        loss2 = np.mean((pred2 - gt2) ** 2)

        print(f"Train: {train_filename}, Test: {test_filename}")
        print("Low interval", loss0)
        print("Mid interval", loss1)
        print("Long interval:", loss2)
        print("Total", loss0 + loss1 + loss2)
        print()