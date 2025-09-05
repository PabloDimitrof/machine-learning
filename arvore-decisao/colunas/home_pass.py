import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/mydata.csv")
home_pass = df["home_pass"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(home_pass, bins=15, edgecolor="black")

ax.set_title("Distribuição de passes do mandante (home_pass)")
ax.set_xlabel("Número de passes")
ax.set_ylabel("Frequência")

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
