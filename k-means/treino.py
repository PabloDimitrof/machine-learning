import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
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

df["attendance"] = (
    df["attendance"].astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", "0").astype(float)
)
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

X_cat = pd.get_dummies(
    df[cat_cols].astype(str).apply(lambda s: s.str.strip()),
    drop_first=False, dtype=int
)

scaler = StandardScaler()
X_num = df[num_cols].copy()
X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_pca)

plt.figure(figsize=(12, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="*", s=220, label="Centroids")
plt.title("K-Means Clustering (Football Matches) - PCA 2D")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()

buf = BytesIO()
plt.savefig(buf, format="png", transparent=True, bbox_inches="tight")
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode("utf-8")
html_img = f'<img src="data:image/png;base64,{img_base64}" alt="KMeans clustering" />'
print(html_img)
