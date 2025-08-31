import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/mydata.csv")
away_pos = df["away_possessions"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(away_pos, bins=15, edgecolor="black")

ax.set_title("Distribuição da posse de bola do visitante (away_possessions)")
ax.set_xlabel("Posse de bola (%)")
ax.set_ylabel("Frequência")

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
