## Introdução
Aplicar **Random Forest** para prever o resultado de partidas da Premier League (`class`: 0=mandante, 1=empate, 2=visitante) usando as ~10 variáveis já utilizadas nos modelos anteriores (estádio, público, posse, passes, chances e IDs dos times). Mantivemos o mesmo padrão do projeto: remoção de colunas com vazamento, limpeza de tipos (ex.: `attendance`), **estratificação** na divisão 70/30 e **sem** usar estatísticas de gols.

## Base de dados
A base de dados utilizada neste projeto contém informações de partidas de futebol, totalizando **1140 linhas** e **40 colunas**. Entre as variáveis estão posse de bola, número de passes e chances criadas.  
A variável alvo escolhida é a coluna **`class`**, que indica o resultado da partida (vitória do mandante, empate ou vitória do visitante).

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


```python exec="0"
--8<-- "docs/arvore-decisao/div.py"
```

## Treinamento do Modelo

O conjunto de **treino** (70%) foi usado para ajustar a floresta; o **teste** (30%) permaneceu isolado para avaliação justa.


=== "Modelo"
    ```python exec="on" html="1"    
    --8<-- "docs/random-forest/treino.py"
    ```
=== "Código"
    ```python exec="0"    
    --8<-- "docs/random-forest/treino.py"
    ```

## Avaliação do Modelo

**Acurácia (teste)**: **0.5673**.  
  Esse desempenho supera os resultados anteriores (Árvore ≈ 0,48; KNN ≈ 0,42–0,51), indicando melhor **generalização** e **robustez** da Random Forest no problema.

- **Importância das variáveis** (top-10):
  1. `away_chances` — 0.2216  
  2. `home_chances` — 0.1160  
  3. `Home Team` — 0.1119  
  4. `Away Team` — 0.1092  
  5. `away_pass` — 0.1039  
  6. `home_pass` — 0.1002  
  7. `attendance` — 0.0867  
  8. `away_possessions` — 0.0556  
  9. `home_possessions` — 0.0517  
  10. `stadium` — 0.0432

**Leituras rápidas**:
- **Chances criadas** (mandante/visitante) e **passes** são os sinais mais relevantes — coerente com dinâmica ofensiva/controle de jogo.  
- Os **IDs dos times** entram forte (efeitos fixos de qualidade/estilo).  
- **Estádio** pesa menos, sugerindo que, após controlar por equipe e métricas do jogo, o local adiciona pouca informação.

## Relatório Final

A **Random Forest** apresentou melhor desempenho que a Árvore de Decisão e o KNN, indicando ganho com **agregação de múltiplas árvores** e menor variância. Ainda assim, prever resultados de futebol segue difícil (classe de **empate** é rara e ruidosa), o que limita a acurácia.

Resultado geral: a RF é a **melhor baseline** até aqui para este conjunto de variáveis, com boa interpretabilidade via importâncias e espaço claro para melhoria com *feature engineering* e ajuste fino.