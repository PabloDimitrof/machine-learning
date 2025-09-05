import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/mydata.csv")
away_pass = df["away_pass"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(away_pass, bins=15, edgecolor="black")

ax.set_title("Distribuição de passes do visitante (away_pass)")
ax.set_xlabel("Número de passes")
ax.set_ylabel("Frequência")

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
