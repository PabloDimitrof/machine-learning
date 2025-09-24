import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = "./src/mydata.csv"
df = pd.read_csv(DATA_PATH)

y_map = {"h": 0, "d": 1, "a": 2}
y = df["class"].astype(str).str.strip().map(y_map).astype(int)

num_cols = [
    "attendance",
    "home_possessions", "away_possessions",
    "home_pass", "away_pass",
    "home_chances", "away_chances",
]
cat_cols = ["stadium", "Home Team", "Away Team"]

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

X_cat = pd.get_dummies(
    df[cat_cols].astype(str).apply(lambda s: s.str.strip()),
    drop_first=False, dtype=int
)
X_num = df[num_cols].copy()

scaler = StandardScaler()
X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
        k_idx = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_idx]
        vals, counts = np.unique(k_labels, return_counts=True)
        return vals[np.argmax(counts)]

knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia (KNN k={knn.k}): {acc:.2f}")
