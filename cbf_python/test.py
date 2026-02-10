import pandas as pd

csv = pd.read_csv("parameters_set.csv")
n_pos = float( csv.loc[csv["ID"] == 4999, "n_pos"].values[0])

print(n_pos)
print(type(n_pos))