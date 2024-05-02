import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import TensorDataset
from torch import nn

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
          nn.Linear(4, 64),
          nn.BatchNorm1d(64),
          nn.ReLU(inplace=True),
          nn.Dropout(0.4),
          nn.Linear(64, num_classes),
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.network(x)
        forgot, forgot_day = x[:,0], x[:,1:]
        return forgot, forgot_day

    def predict_at_t(self, x, t):
        with torch.no_grad():
            forgot, forgot_day = self.forward(x)

        x = self.softmax(forgot_day)
        cdf = torch.cumsum(x, dim=1)

        return cdf[torch.arange(x.size(0)), t.long()]

    def predict(self, x, forgetting_rate=0.71):
        with torch.no_grad():
            forgot, forgot_day = self.forward(x)

        x = self.softmax(forgot_day)
        cdf = torch.cumsum(x, dim=1)
        x = torch.sum((cdf <= forgetting_rate).float(), dim=1)
        x = torch.maximum(x - 1, torch.tensor(1))

        return x, cdf, torch.sigmoid(forgot)

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

for train_filename in ["train.csv", "train_extra.csv"]:
    for test_filename in ["test.csv", "test_extra.csv"]:
        features = ["elapsedDays_{t-1}", "correctSum_{1:t-1}", "incorrectSum_{1:t-1}", "ease_{t-1}"]

        ####################
        # Train model
        ####################

        train_df = pd.read_csv(train_filename)
        test_df = pd.read_csv(test_filename)

        num_classes = int(max(train_df["elapsedDays_{t}"].max(), test_df["elapsedDays_{t}"].max()) + 1 + 1)
        print("num classes", num_classes)

        seed_all(0)

        train_X = torch.from_numpy(train_df[features].values).float()
        train_y = torch.from_numpy(train_df[["recall_{t}", "elapsedDays_{t}"]].values).float()
        train_y[...,0] = 1 - train_y[...,0]
        val_X = torch.from_numpy(test_df[features].values).float()
        val_y = torch.from_numpy(test_df[["recall_{t}", "elapsedDays_{t}"]].values).float()
        val_y[...,0] = 1 - val_y[...,0]
        
        model = Model(num_classes)

        train_dataset = TensorDataset(train_X, train_y.long())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

        val_dataset = TensorDataset(val_X, val_y.float())
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

        ce_loss = nn.CrossEntropyLoss(reduction='none')
        bce_loss = nn.BCEWithLogitsLoss()

        stop = 0
        best_res = 1e5
        while stop < 15:
            model.train()

            train_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
            
                forgot, forgot_day = model(x)
                loss = torch.mean(y[...,0].float() * ce_loss(forgot_day, y[...,1])) + bce_loss(forgot, y[...,0].float())

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                    
            model.eval()

            res = 0
            res2 = 0
            with torch.no_grad():
                for x, y in val_loader:
                    forgot, forgot_day = model(x)
                    forgot_day = torch.argmax(forgot_day, dim=1)
                    forgot = torch.sigmoid(forgot).squeeze()
                    res += torch.mean(y[...,0].float() * (forgot_day - y[...,1]) ** 2)
                    res2 += torch.mean((y[...,0].float() - forgot) ** 2)
            res = torch.sqrt(res / len(val_loader)) + torch.sqrt(res2 / len(val_loader))

            if res < best_res:
                best_res = res
                stop = 0
                torch.save(model.state_dict(), "weight.pt")

            else:
                stop += 1

        model.load_state_dict(torch.load(f"weight.pt"))
        model.eval()

        ####################
        # Test model
        ####################

        test_df = pd.read_csv(test_filename)

        df_features = test_df[features].values
        df_ground_truth = test_df["recall_{t}"]

        pred = model.predict_at_t(val_X, val_y[...,1])

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