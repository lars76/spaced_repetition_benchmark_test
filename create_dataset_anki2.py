import pyzstd
import numpy as np
import pandas as pd
import zipfile
from anki.utils import tmpfile
from anki.collection import Collection
from sklearn.model_selection import StratifiedGroupKFold

ANKI_DECK_FILES = ["deck.apkg"]

def create_extra_data(df, max_days=100):
    # only select all cards where we forgot the card
    masked = df[df['recall_{t}'] == 0]

    # start at the day we know the card was forgotten
    # and add rows up to day 100
    new_rows = []
    for _, row in masked.iterrows():
        for i in range(int(row["elapsedDays_{t}"]), max_days):
            cur_row = row.copy()
            cur_row["elapsedDays_{t}"] = i
            new_rows.append(cur_row)
    new_rows_forgotten = pd.DataFrame(new_rows)

    # only select all cards where we remembered the card
    masked = df[df['recall_{t}'] == 1]

    # stop at the day we know the card was remembered
    new_rows = []
    for _, row in masked.iterrows():
        for i in range(1, int(row["elapsedDays_{t}"])+1):
            cur_row = row.copy()
            cur_row["elapsedDays_{t}"] = i
            new_rows.append(cur_row)
    new_rows_remembered = pd.DataFrame(new_rows)
    concat = pd.concat((new_rows_forgotten, new_rows_remembered))
    concat.sort_values(["cid", "elapsedDays_{t-1}", "elapsedDays_{t}"], inplace=True)

    return concat

####################################
# Load Anki decks
####################################

dfs = []
for i, anki_deck_file in enumerate(ANKI_DECK_FILES):
    with zipfile.ZipFile(anki_deck_file) as zip:
        col_bytes = pyzstd.decompress(zip.read("collection.anki21b"))
    
    colpath = tmpfile(suffix=".anki21")
    with open(colpath, "wb") as buf:
        buf.write(col_bytes)
    col = Collection(colpath)
    
    # A description for the fields is found here:
    # https://github.com/ankidroid/Anki-Android/wiki/Database-Structure
    cols_names = ["id", "cid", "usn", "ease", "ivl", "lastivl", "factor", "time", "type"]
    df = pd.DataFrame(list(col.db.execute(f"SELECT {','.join(cols_names)} FROM revlog")),
                      columns=cols_names)
    df["cid"] = df["cid"].apply(lambda x : anki_deck_file.replace(".apkg", "") + "_" + str(x))
    df.sort_values(["cid", "id"], inplace=True)
    dfs.append(df)
    
df = pd.concat(dfs)

df["cid"] = pd.factorize(df["cid"])[0]
# cram is basically reviewing ahead
df["type"] = df["type"].map({0:"learn", 1:"review", 2:"relearn", 3:"cram"})

# convert timestamp to days
# Anki resets the day at 4:00 am
df["id"] = (pd.to_datetime(df["id"], unit='ms') - pd.Timedelta(hours=4)).dt.date

# Filter the DataFrame to keep only rows where type == "review"
# and the last row where type == "learn"
df_review = df[(df['type'] == 'review') | (df['type'] == 'cram')]
df_learn = df[df['type'] == 'learn'].groupby('cid').tail(1)
df = pd.concat([df_review, df_learn])
df.sort_values(["cid", "id"], inplace=True)

# remove any rows on the same day, keep only the last row
# this avoids elapsedDays == 0
df.drop_duplicates(subset=["id", "cid"], keep='last', inplace=True)

####################################
# Start creating features
####################################

# only now we can compute the elapsed time
df["elapsedDays_{t}"] = df.groupby("cid")["id"].diff(periods=1).dt.total_seconds() / 86400
df["elapsedDays_{t-1}"] = df.groupby("cid")["elapsedDays_{t}"].shift(1)
df["increaseDays"] = df["elapsedDays_{t}"] / df["elapsedDays_{t-1}"]

# remove the "learn" rows, we do not need it for t-1 anymore
df = df[df['type'] == 'review']

# target variable: did we remember the word?
# df["ease"].map({1:"wrong",2:"hard", 3:"ok", 4:"easy"})
df["recall_{t}"] = (df["ease"] != 1).astype("float")
df["recall_{t-1}"] = df.groupby("cid")["recall_{t}"].shift(1)
df["forgetting_{t-1}"] = 1 - df["recall_{t-1}"]

# how easy was the card?
df["ease_{t}"] = df["ease"]
df["ease_{t-1}"] = df.groupby("cid")["ease"].shift(1)

# count the number of correct/incorrect answers up to time t-1
df["correctSum_{1:t-1}"] = df.groupby("cid")["recall_{t-1}"].cumsum()
df["incorrectSum_{1:t-1}"] = df.groupby("cid")["forgetting_{t-1}"].cumsum()

# when the user stopped using Anki for some time, we can have elapsedDays sequences such as [3,90,3]
# this fixes the issue when the user stopped using Anki
# additionally: the bigger the interval, the sparser our knowledge
# we tend to have long-tailed distributions
df = df[df["increaseDays"] <= 7]

df.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)

features = ["incorrectSum_{1:t-1}", "correctSum_{1:t-1}", "ease_{t-1}", "elapsedDays_{t-1}"]
additional_cols = ["cid", "recall_{t}", "elapsedDays_{t}"]
keep_cols = features + additional_cols

df.drop_duplicates(subset=keep_cols, keep='first', inplace=True)
df.reset_index(inplace=True, drop=True)
df = df[keep_cols].astype(int)

####################################
# Splitting the dataset
####################################

X = np.arange(df.shape[0])
y = df["recall_{t}"]
groups = df["cid"]

group_kfold = StratifiedGroupKFold(n_splits=2)

train_index, test_index = next(group_kfold.split(X, y, groups))

train_df = df.iloc[train_index]
train_df.to_csv("train.csv", index=False)
train_df_extra = create_extra_data(train_df)
train_df_extra.to_csv("train_extra.csv", index=False)

test_df = df.iloc[train_index]
test_df.to_csv("test.csv", index=False)
test_df_extra = create_extra_data(test_df)
test_df_extra.to_csv("test_extra.csv", index=False)