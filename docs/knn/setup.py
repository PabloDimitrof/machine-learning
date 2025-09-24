import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1) Carregar sua base
df = pd.read_csv("./src/mydata.csv")

# 2) Alvo (class): h=mandante, d=empate, a=visitante
y_map = {"h": 0, "d": 1, "a": 2}
y = df["class"].astype(str).str.strip().map(y_map).astype(int)

# 3) Seleção de features (≈10, sem vazamento)
num_cols = [
    "attendance",
    "home_possessions", "away_possessions",
    "home_pass", "away_pass",
    "home_chances", "away_chances",
]
cat_cols = ["stadium", "Home Team", "Away Team"]

# 4) Limpezas mínimas para numéricas (ex.: attendance vem como texto com pontuação)
df["attendance"] = (
    df["attendance"]
      .astype(str)
      .str.replace(r"[^0-9]", "", regex=True)
      .replace("", "0")
      .astype(float)
)

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

# 5) One-hot para categóricas; escala para numéricas
X_cat = pd.get_dummies(
    df[cat_cols].astype(str).apply(lambda s: s.str.strip()),
    drop_first=False, dtype=int
)
X_num = df[num_cols].copy()

scaler = StandardScaler()
X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values

# 6) Split (mesma estrutura)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7) PCA só para visualização (2D)
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# 8) KNN (baseline igual)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_2d, y_train)
predictions = knn.predict(X_test_2d)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# 9) Fronteira de decisão (no espaço 2D do PCA)
plt.figure(figsize=(12, 10))
h = 0.05
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_train_2d[:, 0], y=X_train_2d[:, 1], hue=y_train,
                palette="deep", s=100, edgecolor="k", alpha=0.8, legend="full")

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KNN Decision Boundary (Football Matches)")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())