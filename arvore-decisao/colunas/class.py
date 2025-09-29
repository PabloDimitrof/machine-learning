import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/mydata.csv")

class_counts = df["class"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(class_counts.index, class_counts.values, color=["steelblue", "orange", "green"])

ax.set_title("Distribuição dos resultados (class)")
ax.set_xlabel("Resultado (h = mandante, d = empate, a = visitante)")
ax.set_ylabel("Número de partidas")

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
