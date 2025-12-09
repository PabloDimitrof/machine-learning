# Projeto de Machine Learning — Projeto 1

Bem-vindo! Este espaço reúne o **Projeto 1 de Machine Learning** desenvolvido no formato do Prof. Humberto Sandmann, com foco em organização clara, reprodutibilidade e comparação de algoritmos.

## Objetivo Geral
Investigar, treinar e avaliar diferentes algoritmos de *Machine Learning* em um conjunto de dados tabular, passando por todas as etapas do fluxo padrão:
1. **Exploração dos Dados (EDA)**
2. **Pré-processamento**
3. **Divisão Treino/Teste**
4. **Treinamento de Modelos**  
   - Árvore de Decisão  
   - K-Nearest Neighbors (KNN) — versão manual e `scikit-learn`  
   - K-Means (clustering)  
   - Random Forest
5. **Avaliação do Modelo** (acurácia, precisão, recall, F1, matriz de confusão, ROC-AUC)
6. **Relatório Final** com achados e próximos passos

## Base de Dados
- **Dataset principal:** [Premier League Stats](https://www.kaggle.com/datasets/mohamadsallah5/english-premier-league-stats20212024)
- **Alvo supervisionado (quando aplicável):** `quality_high` (derivado de `points ≥ 90`).  
- **Observação:** Em tarefas não supervisionadas (K-Means), o rótulo não é usado no treino; métricas são obtidas via mapeamento posterior (voto majoritário).

## Metodologia (visão resumida)
- **EDA:** descrição das variáveis, distribuição, frequências e possíveis outliers (gráficos específicos por tipo de dado).  
- **Pré-processamento:** limpeza de colunas, conversões de tipo, *label encoding*/*one-hot* para categóricas, padronização quando necessário e **evitar vazamento** (ex.: `points` não entra como *feature* quando gera o alvo).  
- **Split:** 70% treino / 30% teste, **estratificado** pelo alvo.  
- **Modelagem:**  
  - **Árvore de Decisão**: baseline interpretável.  
  - **KNN**: implementação **manual** para didática + versão `scikit-learn` com PCA (visualização da fronteira).  
  - **K-Means**: **k=3** (exploratório), PCA 2D e avaliação via mapeamento cluster→classe.  
  - **Random Forest**: *ensemble* para reduzir variância e melhorar generalização.

## Como Reproduzir
1. Garanta um ambiente com Python 3.10+ e instale:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
