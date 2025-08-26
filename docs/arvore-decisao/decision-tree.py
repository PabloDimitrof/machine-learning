import pandas as pd
import matplotlib.pyplot as plt
import os

# Carregar o dataset
data = pd.read_csv('src/mydata.csv')

# Selecionar características e alvo
print(data.head(1))