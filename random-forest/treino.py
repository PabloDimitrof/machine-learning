import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ===================== Ler base =====================
df = pd.read_csv("./src/mydata.csv")

# ===================== Excluir colunas de vazamento/irrelevantes =====================
df = df.drop(columns=[
    "date", "clock", "links",
    "Goals Home", "Away Goals",
    "home_shots", "away_shots",
    "home_on", "away_on",
    "home_off", "away_off",
    "home_blocked", "away_blocked",
    "home_corners", "away_corners",
    "home_offside", "away_offside",
    "home_tackles", "away_tackles",
    "home_duels", "away_duels",
    "home_saves", "away_saves",
    "home_fouls", "away_fouls",
    "home_yellow", "away_yellow",
    "home_red", "away_red"
], errors="ignore")

# ===================== Limpezas =====================
# stadium em texto -> label encoding
le_stadium = LabelEncoder()
df["stadium"] = le_stadium.fit_transform(df["stadium"].astype(str))

# class: h/d/a -> 0/1/2
df["class"] = df["class"].replace({"h": 0, "d": 1, "a": 2}).astype(int)

# attendance: "12,345" -> 12345
df["attendance"] = (
    df["attendance"]
      .astype(str)
      .str.replace(r"[^0-9]", "", regex=True)
      .replace("", "0")
      .astype(float)
)

# garantir numéricos + imputar medianas nas features
num_cols = [
    "attendance",
    "home_possessions", "away_possessions",
    "home_pass", "away_pass",
    "home_chances", "away_chances"
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    med = df[c].median() if df[c].notna().any() else 0
    df[c] = df[c].fillna(med)

# ===================== Features e alvo =====================
X = df[[
    "stadium", "attendance",
    "Home Team", "Away Team",
    "home_possessions", "away_possessions",
    "home_pass", "away_pass",
    "home_chances", "away_chances"
]].copy()

y = df["class"].copy()

# Alguns times podem ter NA; garantir numérico
X["Home Team"] = pd.to_numeric(X["Home Team"], errors="coerce").fillna(-1).astype(int)
X["Away Team"] = pd.to_numeric(X["Away Team"], errors="coerce").fillna(-1).astype(int)

# ===================== Split (70/30) com estratificação =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=27, stratify=y
)

# ===================== Random Forest =====================
rf = RandomForestClassifier(
    n_estimators=100,      # número de árvores
    max_depth=5,          # profundidade máxima (controle de overfitting)
    max_features='sqrt',  # nº de features por split
    random_state=27,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ===================== Avaliação =====================
pred = rf.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.4f}")

# Importância das features
feat_imp = pd.DataFrame({
    "Feature": rf.feature_names_in_,
    "Importância": rf.feature_importances_
}).sort_values("Importância", ascending=False)

print("<br>Importância das Features:")
print(feat_imp.to_html(index=False))
