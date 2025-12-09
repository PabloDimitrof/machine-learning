import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/mydata.csv")
away_chances = df["away_chances"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(away_chances, bins=15, edgecolor="black")

ax.set_title("Distribuição de chances do visitante (away_chances)")
ax.set_xlabel("Número de chances")
ax.set_ylabel("Frequência")

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
