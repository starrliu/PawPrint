import pandas as pd

df = pd.read_csv("data/gt.csv")

df = df[:10000]

df.to_csv("data/gt_10000.csv", index=False)