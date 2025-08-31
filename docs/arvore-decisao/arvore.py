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

# Variáveis independentes (features)
x = df[[
    "stadium", "attendance",
    "Home Team", "Away Team",
    "home_possessions", "away_possessions",
    "home_pass", "away_pass",
    "home_chances", "away_chances"
]]

# Variável dependente (alvo)
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=27, stratify=y)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Avaliar o modelo
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisão da Validação: {accuracy:.2f}")

feature_importance = pd.DataFrame({
    'Feature': classifier.feature_names_in_,
    'Importância': classifier.feature_importances_
})
print("<br>Importância das Features:")
print(feature_importance.sort_values(by='Importância', ascending=False).to_html())

plt.figure(figsize=(20, 10))
tree.plot_tree(classifier, max_depth=5, fontsize=10)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())    
