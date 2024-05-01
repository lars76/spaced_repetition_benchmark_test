import numpy as np
import pandas as pd
import ebisu

for train_filename in ["train.csv"]:#, "train_extra.csv"
    for test_filename in ["test.csv"]: #"test.csv", 
        features = ["elapsedDays_{t-1}", "correctSum_{1:t-1}", "incorrectSum_{1:t-1}", "ease_{t-1}"]

        ####################
        # Test model
        ####################

        test_df = pd.read_csv(test_filename)

        pred = []
        grouped = test_df.groupby("cid")
        for _, rows in grouped:
            model = (2., 2., 24.)
            for _, row in rows.iterrows():
                recall = ebisu.predictRecall(model,
                                             row['elapsedDays_{t}'] / 24,
                                             exact=True)
                pred.append(recall)

                model = ebisu.updateRecall(model,
                                           row['recall_{t}'],
                                           1,
                                           row['elapsedDays_{t}'] / 24)

        pred = np.array(pred)

        df_features = test_df[features].values
        df_ground_truth = test_df["recall_{t}"]

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