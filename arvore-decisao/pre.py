import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df = pd.read_csv("./src/mydata.csv")

# Excluir as colunas não desejadas
df = df.drop(columns= ["date", "clock", "links", "Goals Home", "Away Goals", "home_shots", "away_shots", "home_on", "away_on",
                        "home_off", "away_off", "home_blocked", "away_blocked", "home_corners", "away_corners", 
                        "home_offside", "away_offside", "home_tackles", "away_tackles", "home_duels", "away_duels",
                        "home_saves", "away_saves", "home_fouls", "away_fouls", "home_yellow", "away_yellow",
                        "home_red", "away_red"])

# Label encoding dos estadios em texto
df["stadium"] = label_encoder.fit_transform(df["stadium"])

# Transformar resultado do jogo times em números
df["class"] = df["class"].replace({'h':0, 'd': 1, 'a':2})

# Transforma o públido de string para número
df["attendance"] = df["attendance"].str.replace(',', '').astype(int)

print(df.sample(frac=.01).to_markdown(index=False))