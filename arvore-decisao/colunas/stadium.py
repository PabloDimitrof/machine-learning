import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/mydata.csv")

stadiums = (
    df["stadium"]
      .astype(str)
      .fillna("Desconhecido")
      .str.strip()
)

counts = stadiums.value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(counts.index, counts.values)

ax.set_title("Frequência de partidas por estádio")
ax.set_xlabel("Estádio")
ax.set_ylabel("Número de partidas")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
