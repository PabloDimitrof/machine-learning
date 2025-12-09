import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/mydata.csv")

df["attendance"] = df["attendance"].astype(str).str.replace(r"[^0-9]", "", regex=True).astype(float)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["attendance"], bins=15, color="steelblue", edgecolor="black")

ax.set_title("Distribuição do Público (attendance)")
ax.set_xlabel("Número de torcedores")
ax.set_ylabel("Frequência")

plt.xticks(rotation=15)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
