import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./src/mydata.csv")

print(df.sample(frac=.01).to_markdown(index=False))