import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/mydata.csv")

counts = (
    df["Away Team"]
      .astype("Int64")
      .value_counts()
      .sort_index()
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(counts.index.astype(str), counts.values)

ax.set_title("Frequência de partidas por time visitante (Away Team)")
ax.set_xlabel("ID do time (visitante)")
ax.set_ylabel("Número de partidas")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
