import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("./src/mydata.csv")

num_cols = [
    "attendance",
    "home_possessions", "away_possessions",
    "home_pass", "away_pass",
    "home_chances", "away_chances",
]
cat_cols = ["stadium", "Home Team", "Away Team"]
target = "class"

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

scaler = StandardScaler()
X_num = df[num_cols].copy()
X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values
y_map = {"h": 0, "d": 1, "a": 2}
y = df[target].astype(str).str.strip().map(y_map).astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_train_pca)

cluster_map = {}
classes = np.unique(y_train)
for c in np.unique(labels):
    mask = labels == c
    if mask.sum() == 0:
        cluster_map[c] = classes[0]
        continue
    counts = np.bincount(y_train[mask], minlength=classes.max()+1)
    majority = counts.argmax()
    cluster_map[c] = majority

y_pred_train = np.array([cluster_map[c] for c in labels])
acc = accuracy_score(y_train, y_pred_train)
cm = confusion_matrix(y_train, y_pred_train, labels=np.sort(classes))

cm_df = pd.DataFrame(
    cm,
    index=[f"Classe Real {cls}" for cls in np.sort(classes)],
    columns=[f"Classe Pred {cls}" for cls in np.sort(classes)]
)

print(f"Acurácia: {acc*100:.2f}%")
print("<br>Matriz de Confusão:")
print(cm_df.to_html(index=True))
