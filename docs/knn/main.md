## Introdução
O objetivo desta etapa foi aplicar o algoritmo *K-Nearest Neighbors* (KNN) sobre a base de dados de partidas de futebol, utilizando as bibliotecas `pandas`, `numpy`, `matplotlib` e `scikit-learn`.  
Diferente da árvore de decisão, o KNN classifica novas observações com base na proximidade de exemplos já conhecidos.  
A proposta é avaliar como variáveis como estádio, público, posse de bola, passes e chances criadas podem auxiliar na previsão do resultado da partida (`class`).  
Foram desenvolvidas duas abordagens: uma implementação manual, para consolidar a compreensão do funcionamento do método, e outra com a biblioteca *scikit-learn*, permitindo comparação de resultados e visualização da fronteira de decisão por meio do PCA.

## Base de dados
A base de dados utilizada neste projeto contém informações de partidas de futebol, totalizando **1140 linhas** e **40 colunas**. Entre as variáveis estão posse de bola, número de passes e chances criadas.  
A variável alvo escolhida é a coluna **`class`**, que indica o resultado da partida (vitória do mandante, empate ou vitória do visitante), sendo o objeto da classificação pelo algoritmo de árvore de decisão.

## Exploração dos Dados
A seguir foi feita uma análise do significado e composição de cada coluna presente na base, com a finalidade de identificar possíveis problemas a serem tratados posteriormente. As visualizações e estatísticas descritivas ajudam a compreender a natureza dos dados e orientar decisões de pré-processamento e modelagem.

=== "class"

    A coluna `class` é a variável alvo do projeto e representa o resultado da partida: vitória do mandante (*"h"*), empate (*"d"*) ou vitória do visitante (*"a"*). 
    Trata-se de uma variável categórica com três possíveis valores, sendo o objeto de classificação do modelo de árvore de decisão. 
    A análise exploratória dessa coluna é essencial para observar o balanceamento do conjunto de dados, isto é, se há proporções semelhantes ou discrepantes entre os resultados possíveis.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/class.py"
    ```

=== "attendance"

    A coluna `attendance` representa o público presente em cada partida, um indicador de contexto do jogo que pode refletir fatores como mando de campo, relevância do confronto e engajamento da torcida. Em bases de futebol, o público tende a variar bastante entre estádios e rodadas, podendo apresentar assimetria (jogos muito cheios em arenas grandes) e valores atípicos (clássicos, finais).

    Do ponto de vista analítico, é uma variável contínua útil para observar a distribuição de torcedores ao longo das partidas e investigar relações com o resultado (`class`).

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/attendance.py"
    ```

=== "stadium"

    A coluna `stadium` identifica o estádio onde a partida foi disputada. Trata-se de uma variável categórica, associada ao contexto e à capacidade do local, podendo refletir fatores como mando de campo, perfil da torcida, relevo do gramado e até particularidades logísticas. 
    Na análise exploratória, é útil observar a **frequência de jogos por estádio**, verificando balanceamento da amostra (quais arenas têm mais/menos partidas) e possíveis vieses (ex.: concentração em poucos estádios). 
    Relações mais profundas com o resultado (`class`) podem ser investigadas em etapas seguintes, mas aqui focamos em entender a **composição dessa variável** no conjunto de dados.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/stadium.py"
    ```

=== "home_possessions"

    A coluna `home_possessions` representa a porcentagem de posse de bola do time mandante em cada partida. 
    Trata-se de uma variável numérica contínua, normalmente variando entre 30% e 70% na maioria dos jogos, podendo indicar estilos de jogo (times que mantêm a bola ou que jogam mais reativamente). 
    A análise exploratória dessa variável permite observar a distribuição da posse de bola entre os mandantes, identificar valores atípicos e comparar a média do controle de jogo ao longo das rodadas. 
    Posteriormente, poderá ser interessante relacionar essa posse com o resultado final (`class`) para verificar padrões.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/home_possessions.py"
    ```

=== "away_possessions"

    A coluna `away_possessions` representa a porcentagem de posse de bola do time visitante em cada partida. 
    Por ser uma variável numérica contínua, sua distribuição ajuda a observar o comportamento dos visitantes em termos de controle de jogo, identificar valores atípicos e comparar a tendência média de posse fora de casa. 
    Em etapas posteriores, pode ser relacionada ao resultado (`class`) para investigar padrões de desempenho como visitante.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/away_possessions.py"
    ```

=== "Home Team"

    A coluna `Home Team` identifica o time mandante na partida. Embora esteja codificada numericamente no dataset, sua natureza é categórica (IDs de times). 
    Na análise exploratória, é útil observar a frequência de jogos por mandante, verificando o balanceamento da amostra entre os times que atuam em casa e possíveis concentrações. 
    Relações com o resultado (`class`) podem ser exploradas depois; aqui focamos em entender a composição dessa variável.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/home_team.py"
    ```

=== "Away Team"

    A coluna `Away Team` identifica o time visitante na partida. Apesar de codificada como número, sua natureza é categórica (IDs de times). 
    Na análise exploratória, observar a frequência de jogos por visitante ajuda a avaliar o balanceamento da amostra e possíveis concentrações de partidas em determinados clubes.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/away_team.py"
    ```

=== "home_pass"

    A coluna `home_pass` indica o número de passes realizados pelo time mandante em cada partida. 
    É uma variável numérica contínua que ajuda a caracterizar o estilo de jogo do mandante, podendo variar bastante entre equipes mais ou menos dependentes da posse de bola. 
    Na análise exploratória, observar a distribuição dos passes permite identificar médias, dispersão e valores atípicos.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/home_pass.py"
    ```

=== "away_pass"

    A coluna `away_pass` indica o número de passes realizados pelo time visitante em cada partida. 
    Sendo uma variável numérica contínua, sua distribuição mostra como os visitantes se comportam em termos de construção de jogadas e controle de posse fora de casa. 
    A análise exploratória ajuda a entender a média de passes, variações entre os jogos e eventuais valores extremos.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/away_pass.py"
    ```

=== "home_chances"

    A coluna `home_chances` representa a quantidade de chances de gol criadas pelo time mandante durante a partida. 
    É uma variável numérica discreta que indica o nível de ofensividade da equipe jogando em casa. 
    A análise exploratória permite identificar a frequência de jogos com poucas ou muitas oportunidades e verificar a dispersão desse tipo de estatística.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/home_chances.py"
    ```

=== "away_chances"

    A coluna `away_chances` indica a quantidade de chances de gol criadas pelo time visitante durante a partida. 
    É uma variável numérica discreta que ajuda a compreender a ofensividade dos times jogando fora de casa. 
    A análise exploratória mostra como os visitantes se comportam em termos de criação de oportunidades, permitindo identificar padrões de equilíbrio ou diferenças marcantes em relação aos mandantes.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/away_chances.py"
    ```


## Pré-processamento

Após a exploração inicial da base, foram aplicados procedimentos de pré-processamento para preparar os dados para o treinamento do modelo.  
Entre as etapas realizadas estão:

- **Conversão de tipos**: variáveis originalmente em texto com valores numéricos (como `attendance`) foram transformadas em formato numérico.  
- **Tratamento de valores categóricos**: colunas como `stadium`, `Home Team` e `Away Team` foram mantidas como categóricas, sendo posteriormente convertidas em variáveis numéricas por meio de técnicas de codificação.  
- **Remoção de colunas irrelevantes ou redundantes**: colunas de identificação e de tempo (`date`, `clock`, `links`), bem como estatísticas que representam vazamento de informação do resultado (ex.: `Goals Home`, `Away Goals`), foram descartadas do conjunto de treino.  
- **Separação entre features e target**: as variáveis explicativas (`X`) foram definidas a partir de aproximadamente dez colunas relevantes da base, enquanto a variável alvo (`y`) é a coluna `class`.  
- **Divisão em treino e teste**: o conjunto de dados foi dividido em duas partes, garantindo estratificação do alvo para manter o equilíbrio das classes.

Com essas etapas, os dados foram organizados de forma consistente, reduzindo ruídos e preparando a base para a etapa seguinte: o treinamento do modelo de árvore de decisão.

=== "Base preparada"

    ```python exec="1"
    --8<-- "docs/arvore-decisao/pre.py"
    ```
=== "code"

    ```python exec="0"
    --8<-- "docs/arvore-decisao/pre.py"
    ```
=== "Base original"

    ```python exec="1"
    --8<-- "docs/arvore-decisao/base0.py"
    ```

## Divisão dos Dados

Com a base pré-processada, realizou-se a divisão entre conjuntos de **treinamento** e **teste**.  
O objetivo dessa etapa é garantir que o modelo seja avaliado em dados que ele nunca viu durante o treinamento, permitindo uma medida mais confiável de sua capacidade de generalização.  

Foi utilizada a função `train_test_split` da biblioteca *scikit-learn*, com os seguintes critérios:  
- **70% dos dados** destinados ao treinamento, para que o modelo aprenda os padrões da base;  
- **30% dos dados** destinados ao teste, para avaliar o desempenho em novos exemplos;  
- **Estratificação pelo alvo (`class`)**, garantindo que a proporção entre vitórias do mandante, empates e vitórias do visitante fosse mantida em ambos os conjuntos;  
- **Random State** fixado, assegurando reprodutibilidade na divisão.  

Dessa forma, o conjunto de treinamento foi usado para ajustar os parâmetros da árvore de decisão, enquanto o conjunto de teste serviu para medir a precisão e robustez do modelo.

```python exec="0"
--8<-- "docs/arvore-decisao/div.py"
```

## Treinamento do Modelo

Nesta seção foi implementado o algoritmo KNN de forma manual, a partir do zero, para consolidar o entendimento do funcionamento do método.  
A implementação considera a distância euclidiana entre os pontos, identifica os vizinhos mais próximos e atribui a classe com maior frequência.  
Esse exercício é importante para compreender a lógica por trás do KNN antes de utilizar bibliotecas prontas.

=== "Modelo"
    ```python exec="on" html="1"    
    --8<-- "docs/knn/modelo.py"
    ```
=== "Código"
    ```python exec="0"    
    --8<-- "docs/knn/modelo.py"
    ```

## Usando Scikit-Learn

Nesta seção foi implementado o algoritmo KNN de forma manual, a partir do zero, para consolidar o entendimento do funcionamento do método.  
A implementação considera a distância euclidiana entre os pontos, identifica os vizinhos mais próximos e atribui a classe com maior frequência.  
Esse exercício é importante para compreender a lógica por trás do KNN antes de utilizar bibliotecas prontas.

=== "Resultado"
    ```python exec="on" html="1"    
    --8<-- "docs/knn/setup.py"
    ```
=== "Código"
    ```python exec="0"    
    --8<-- "docs/knn/setup.py"
    ```

## Avaliação do Modelo

Após o treinamento do algoritmo KNN, o modelo foi avaliado no conjunto de teste, obtendo resultados de acurácia em torno de **0.42** (usando scikit-learn) e **0.51** (na implementação manual).  
Esses valores mostram que o modelo consegue acertar parte das previsões, mas ainda apresenta uma performance limitada.  

Esse resultado deve ser interpretado considerando a **natureza do problema**: prever o resultado de uma partida de futebol é uma tarefa complexa, com alto grau de incerteza.  
Mesmo dispondo de estatísticas como posse de bola, passes, chances criadas, estádio e público, o desfecho de um jogo depende também de fatores externos como arbitragem, lesões, clima, motivação e até elementos de acaso.  
Portanto, ainda que houvesse mais dados disponíveis, a previsão exata continuaria sendo incerta.  

A figura abaixo mostra a **fronteira de decisão do KNN** no espaço bidimensional reduzido pelo PCA.  
As regiões coloridas representam as áreas de influência de cada classe, enquanto os pontos correspondem às partidas reais.

É possível observar uma forte **sobreposição entre classes**, com pontos de diferentes resultados distribuídos de forma bastante misturada. Isso reforça as limitações do modelo:

- **Baixa capacidade preditiva**: as variáveis utilizadas ajudam a descrever o contexto, mas não criam fronteiras claras para separar vitórias, empates e derrotas.  
- **Balanceamento do alvo**: os empates, menos frequentes, tendem a ser mal classificados, prejudicando a acurácia global.  
- **Sensibilidade do algoritmo**: o KNN depende fortemente da escala, da escolha de `k` e da representação dos dados, o que gera variações nos resultados.  
- **Complexidade do domínio**: a imprevisibilidade do futebol limita naturalmente a precisão que qualquer modelo pode alcançar.  

Portanto, as acurácias obtidas refletem tanto as limitações do KNN quanto a complexidade do fenômeno esportivo. Mais do que “acertar resultados”, o exercício evidencia os desafios de aplicar *Machine Learning* em cenários reais de alta incerteza.

## Relatório Final

O projeto teve como objetivo aplicar o algoritmo KNN sobre uma base de partidas de futebol, explorando seu potencial no contexto esportivo.  

A análise foi conduzida em etapas bem definidas:

- **Exploração dos Dados (EDA)**: foram selecionadas e analisadas cerca de dez variáveis relevantes, incluindo `stadium`, `attendance`, `Home Team`, `Away Team`, `home_possessions`, `away_possessions`, `home_pass`, `away_pass`, `home_chances` e `away_chances`.  
- **Pré-processamento**: colunas irrelevantes (como IDs e dados de tempo) e estatísticas ligadas diretamente ao resultado (como gols) foram removidas. Variáveis categóricas foram transformadas em numéricas via *One-Hot Encoding* e variáveis contínuas foram padronizadas.  
- **Divisão dos Dados**: o conjunto foi separado em treino (70%) e teste (30%), preservando a proporção entre vitórias, empates e derrotas através da estratificação.  
- **Treinamento e Avaliação**: o KNN foi aplicado em duas versões — manual e com *scikit-learn* — alcançando acurácias de **0.42–0.51**, valores que evidenciam desempenho limitado, mas condizente com a natureza do problema.  

A avaliação trouxe alguns aprendizados importantes:
- O KNN é sensível à preparação dos dados e à escolha de hiperparâmetros, o que explica diferenças entre implementações.  
- As variáveis utilizadas caracterizam parcialmente os jogos, mas não são suficientes para prever com exatidão seus resultados.  
- A visualização da fronteira de decisão mostrou que as classes apresentam alta sobreposição, o que dificulta a separação clara.  
- O futebol, por ser um evento com forte componente de imprevisibilidade, impõe limites naturais ao desempenho de qualquer modelo.  

### Conclusão

Assim como na árvore de decisão, a experiência com o KNN reforça a importância de interpretar os resultados com cautela.  
A acurácia próxima de 0.5 não deve ser vista apenas como limitação do algoritmo, mas como reflexo da complexidade do domínio esportivo.  

O projeto destacou a relevância da análise exploratória, do cuidado com o pré-processamento e da avaliação crítica dos modelos. Além disso, abre espaço para trabalhos futuros que explorem variáveis adicionais (desempenho histórico dos times, estatísticas de jogadores, forma recente, fatores externos) e ajustes de hiperparâmetros do KNN para buscar melhorias.  

Mais do que prever com exatidão, este trabalho evidencia o potencial e os limites do uso de algoritmos de aprendizado de máquina em contextos reais, nos quais a incerteza é parte inerente do fenômeno analisado.
